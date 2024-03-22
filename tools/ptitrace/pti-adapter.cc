#include "pti-adapter.h"
#include "utilities.h"
#include <pti_view.h>
#include <pti_version.h>
#include <ctime>
#include <fstream>
#include <new>
#include <cstddef>
#include <iostream>
#include <nlohmann/json.hpp>

constexpr std::size_t kDefaultViewRecords = 5'000;
constexpr std::size_t kBiggestRecordSize = sizeof(pti_view_record_kernel);
constexpr std::size_t kDefaultBufSize = kDefaultViewRecords * kBiggestRecordSize;
constexpr auto kDefaultAlignment = std::align_val_t{1};

struct PtiSettings {
 public:
  inline static auto& Instance() {
    static PtiSettings settings{};
    return settings;
  }

  PtiSettings() {
    // TODO(ms): Hard coded!!
    //
    chrome_log_["schemaVersion"] = 1;
    chrome_log_["traceName"] = "my_trace.json";
    chrome_log_["displayTimeUnit"] = "ns";
    chrome_log_["otherData"]["app_version"] = "0.0.1";
    chrome_log_["otherData"]["pti_version"] = ptiVersionString();

    auto start_trace_obj = nlohmann::json::object();
    start_trace_obj["ph"] = "M";
    start_trace_obj["name"] = "dpc_gemm";
    start_trace_obj["pid"] = utilities::GetPid();
    start_trace_obj["args"]["name"]  = "dpc_gemm";

    auto start_time_trace_obj = nlohmann::json::object();
    start_time_trace_obj["ph"] = "M";
    start_time_trace_obj["name"] = "start_time";
    start_time_trace_obj["pid"] = utilities::GetPid();
    start_time_trace_obj["args"]["CLOCK_MONOTONIC_RAW"]  = utilities::GetTime(CLOCK_MONOTONIC_RAW);
    start_time_trace_obj["args"]["CLOCK_MONOTONIC"]  = utilities::GetTime(CLOCK_MONOTONIC);
    start_time_trace_obj["args"]["CLOCK_REALTIME"]  = utilities::GetTime(CLOCK_REALTIME);

    chrome_log_["traceEvents"].push_back(start_trace_obj);
    chrome_log_["traceEvents"].push_back(start_time_trace_obj);
  }

  void AddRecord(pti_view_kind view, pti_view_record_base* base) {
    if(!base) {
      return;
    }
    auto trace_obj = nlohmann::json::object();
    switch (view) {
      case pti_view_kind::PTI_VIEW_INVALID: {
        throw std::runtime_error("Found Invalid Record");
        break;
      }
      case pti_view_kind::PTI_VIEW_SYCL_RUNTIME_CALLS: {
        auto* rec = reinterpret_cast<pti_view_record_sycl_runtime*>(base);
        trace_obj["ph"] = "X";
        trace_obj["pid"] = rec->_process_id;
        trace_obj["tid"] = rec->_thread_id;
        trace_obj["name"] = rec->_name;
        trace_obj["ts"] = rec->_start_timestamp;
        trace_obj["dur"] = (rec->_end_timestamp - rec->_start_timestamp);
        trace_obj["args"]["correlation_id"] = rec->_correlation_id;
        chrome_log_["traceEvents"].push_back(trace_obj);
        break;
      }
      case pti_view_kind::PTI_VIEW_DEVICE_GPU_KERNEL: {
        auto* rec = reinterpret_cast<pti_view_record_kernel*>(base);
        trace_obj["ph"] = "X";
        trace_obj["pid"] = utilities::GetPid();
        trace_obj["tid"] = rec->_thread_id;
        trace_obj["name"] = rec->_name;
        trace_obj["ts"] = rec->_start_timestamp;
        trace_obj["dur"] = (rec->_end_timestamp - rec->_start_timestamp);
        trace_obj["args"]["id"] = rec->_kernel_id;
        trace_obj["args"]["correlation_id"] = rec->_correlation_id;
        chrome_log_["traceEvents"].push_back(trace_obj);
        break;
      }
      case pti_view_kind::PTI_VIEW_DEVICE_GPU_MEM_COPY: {
        auto* rec = reinterpret_cast<pti_view_record_memory_copy*>(base);
        trace_obj["ph"] = "X";
        trace_obj["pid"] = utilities::GetPid();
        trace_obj["tid"] = rec->_thread_id;
        trace_obj["name"] = rec->_name;
        trace_obj["ts"] = rec->_start_timestamp;
        trace_obj["dur"] = (rec->_end_timestamp - rec->_start_timestamp);
        trace_obj["args"]["id"] = rec->_mem_op_id;
        trace_obj["args"]["correlation_id"] = rec->_correlation_id;
        trace_obj["args"]["bytes_copied"] = rec->_bytes;
        chrome_log_["traceEvents"].push_back(trace_obj);
        break;
      }
      case pti_view_kind::PTI_VIEW_DEVICE_GPU_MEM_FILL: {
        auto* rec = reinterpret_cast<pti_view_record_memory_fill*>(base);
        trace_obj["ph"] = "X";
        trace_obj["pid"] = utilities::GetPid();
        trace_obj["tid"] = rec->_thread_id;
        trace_obj["name"] = rec->_name;
        trace_obj["ts"] = rec->_start_timestamp;
        trace_obj["dur"] = (rec->_end_timestamp - rec->_start_timestamp);
        trace_obj["args"]["id"] = rec->_mem_op_id;
        trace_obj["args"]["correlation_id"] = rec->_correlation_id;
        trace_obj["args"]["value_for_set"] = rec->_value_for_set;
        trace_obj["args"]["bytes_copied"] = rec->_bytes;
        chrome_log_["traceEvents"].push_back(trace_obj);
        break;
      }
      default:
        break;
    }
  }

  virtual ~PtiSettings() {
    try  {
      std::ofstream result_file("my_trace.json");
      result_file << chrome_log_;
    } catch (...) {
      std::cerr << "Failed to write to file" << '\n';
    }
  }

  constexpr void SetMaxNumberOfRecordsInBuffer(std::size_t num_records) {
    num_view_records_ = num_records;
    buf_size_ = num_view_records_ * kBiggestRecordSize;
  }

  constexpr void SetBufferAlignment(std::align_val_t alignment) { align_ = alignment; }

  inline void CreateBuffer(unsigned char** buffer, std::size_t* buf_size) {
    try {
      *buffer = static_cast<unsigned char*>(::operator new(buf_size_, align_));
      *buf_size = buf_size_;
      if (!*buffer) {
        std::cerr << "Buf Alloc Failed: " << '\n';
        std::abort();
      }
    } catch (const std::bad_alloc& e) {
      std::cerr << "Buf Alloc Failed: " << e.what() << '\n';
      std::abort();
    }
  }

  inline void DeleteBuffer(unsigned char* buffer) { ::operator delete(buffer); }

 private:
  std::size_t num_view_records_ = kDefaultViewRecords;
  std::size_t buf_size_ = kDefaultBufSize;
  std::align_val_t align_ = kDefaultAlignment;
  nlohmann::json chrome_log_ = nlohmann::json::object();
};

constexpr void CheckPtiReturnValue(pti_result result) {
  if (result != pti_result::PTI_SUCCESS) {
    throw std::runtime_error("bad call to pti");
  }
}

void BufferRequested(unsigned char** buffer, std::size_t* buf_size) {
  std::cout << "Buffer requested" << '\n';
  PtiSettings::Instance().CreateBuffer(buffer, buf_size);
}

void BufferReturned(unsigned char* buffer, std::size_t buf_size, std::size_t valid_buf_size) {
  std::cout << "Buffer returned" << '\n';
  if (!buffer || !valid_buf_size || !buf_size) {
    if (valid_buf_size) {
      PtiSettings::Instance().DeleteBuffer(buffer);
    }
    return;
  }
  pti_view_record_base* ptr = nullptr;
  while (true) {
    auto buf_status = ptiViewGetNextRecord(buffer, valid_buf_size, &ptr);
    if (buf_status == pti_result::PTI_STATUS_END_OF_BUFFER) {
      std::cout << "Reached End of buffer" << '\n';
      break;
    }
    if (buf_status != pti_result::PTI_SUCCESS) {
      std::cout << "bad record found" << '\n';
      break;
    }
    PtiSettings::Instance().AddRecord(ptr->_view_kind, ptr);
  }
  PtiSettings::Instance().DeleteBuffer(buffer);
}

template <typename... E>
inline void EnableViews(E... view_kinds) {
  ([&] { CheckPtiReturnValue(ptiViewEnable(view_kinds)); }(), ...);
}

template <typename... E>
inline void DisableViews(E... view_kinds) {
  ([&] { CheckPtiReturnValue(ptiViewDisable(view_kinds)); }(), ...);
}

class TracingLifeTimeTracker {
public:
static inline auto& Instance() {
  static TracingLifeTimeTracker tracker = {};
  return tracker;
}

bool Init() {
  return true;
}

TracingLifeTimeTracker() {
  CheckPtiReturnValue(ptiViewSetCallbacks(BufferRequested, BufferReturned));
  EnableViews(PTI_VIEW_DEVICE_GPU_KERNEL, PTI_VIEW_DEVICE_GPU_MEM_COPY, PTI_VIEW_DEVICE_GPU_MEM_FILL, PTI_VIEW_SYCL_RUNTIME_CALLS);
}

~TracingLifeTimeTracker() {
  DisableViews(PTI_VIEW_DEVICE_GPU_KERNEL, PTI_VIEW_DEVICE_GPU_MEM_COPY, PTI_VIEW_DEVICE_GPU_MEM_FILL, PTI_VIEW_SYCL_RUNTIME_CALLS);
  CheckPtiReturnValue(ptiFlushAllViews());
}
};

class GlobalPtiInitializer {
 public:
  inline static bool Initialize() {
    if(utilities::GetEnv(Env::kActivatePti.begin()) == "1") {
      return TracingLifeTimeTracker::Instance().Init();
    }
    return false;
  }

  inline static bool result_ = Initialize();
};

void PtiAdapterInit() {
  utilities::SetEnv(Env::kActivatePti.begin(), "1");
}

