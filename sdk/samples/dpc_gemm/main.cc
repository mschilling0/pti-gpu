//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#include <string.h>

#include <new>
#include <memory>
#include <sycl/sycl.hpp>
#include <cstdlib>
#include <memory>

#include "pti_view.h"
#include "utils.h"
#include "samples_utils.h"

#define A_VALUE 0.128f
#define B_VALUE 0.256f
#define MAX_EPS 1.0e-4f

#define PTI_CHECK_RETURN(X) do \
  { \
    if (X != pti_result::PTI_SUCCESS) { \
      throw std::runtime_error("Failed call to PTI"); \
    } \
  } while(0) \

void StartTracing() {
  PTI_CHECK_RETURN(ptiViewEnable(PTI_VIEW_DEVICE_GPU_KERNEL));
  PTI_CHECK_RETURN(ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_FILL));
  PTI_CHECK_RETURN(ptiViewEnable(PTI_VIEW_DEVICE_GPU_MEM_COPY));
}

void StopTracing() {
  PTI_CHECK_RETURN(ptiViewDisable(PTI_VIEW_DEVICE_GPU_KERNEL));
  PTI_CHECK_RETURN(ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_FILL));
  PTI_CHECK_RETURN(ptiViewDisable(PTI_VIEW_DEVICE_GPU_MEM_COPY));
}

static float Check(const std::vector<float> &a, float value) {
  PTI_ASSERT(value > MAX_EPS);

  float eps = 0.0f;
  for (size_t i = 0; i < a.size(); ++i) {
    eps += fabs((a[i] - value) / value);
  }

  return eps / a.size();
}

void GEMM(const float *a, const float *b, float *c, unsigned size,
          sycl::id<2> id) {
  int i = id.get(0);
  int j = id.get(1);
  float sum = 0.0f;
  for (unsigned k = 0; k < size; ++k) {
    sum += a[i * size + k] * b[k * size + j];
  }
  c[i * size + j] = sum;
}

static float RunAndCheck(sycl::queue queue, const std::vector<float> &a,
                         const std::vector<float> &b, std::vector<float> &c,
                         unsigned size, float expected_result, std::vector<double>* kernel_times) {
  PTI_ASSERT(size > 0);
  PTI_ASSERT(a.size() == size * size);
  PTI_ASSERT(b.size() == size * size);
  PTI_ASSERT(c.size() == size * size);

  double time = 0.0;

  try {
    sycl::buffer<float, 1> a_buf(a.data(), a.size());
    sycl::buffer<float, 1> b_buf(b.data(), b.size());
    sycl::buffer<float, 1> c_buf(c.data(), c.size());

    sycl::event event = queue.submit([&](sycl::handler &cgh) {
      auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
      auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
      auto c_acc = c_buf.get_access<sycl::access::mode::write>(cgh);

      cgh.parallel_for<class __GEMM>(
          sycl::range<2>(size, size), [=](sycl::id<2> id) {
            auto a_acc_ptr = a_acc.get_multi_ptr<sycl::access::decorated::no>();
            auto b_acc_ptr = b_acc.get_multi_ptr<sycl::access::decorated::no>();
            auto c_acc_ptr = c_acc.get_multi_ptr<sycl::access::decorated::no>();
            GEMM(a_acc_ptr.get(), b_acc_ptr.get(), c_acc_ptr.get(), size, id);
          });
    });
    queue.wait_and_throw();

    auto start =
        event.get_profiling_info<sycl::info::event_profiling::command_start>();
    auto end =
        event.get_profiling_info<sycl::info::event_profiling::command_end>();
    time = static_cast<double>(end - start) / NSEC_IN_SEC;
  } catch (const sycl::exception& e) {
    std::cout << "[ERROR] " << e.what() << std::endl;
    throw;
  }

  if(kernel_times) {
    kernel_times->push_back(time);
  }

  std::cout << "Matrix multiplication time: " << time << " sec" << std::endl;

  return Check(c, expected_result);
}

static void Compute(sycl::queue queue, const std::vector<float> &a,
                    const std::vector<float> &b, std::vector<float> &c,
                    unsigned size, unsigned repeat_count,
                    float expected_result, std::vector<double>* kernel_times) {
  for (unsigned i = 0; i < repeat_count; ++i) {
    float eps = RunAndCheck(queue, a, b, c, size, expected_result, kernel_times);
    std::cout << "Results are " << ((eps < MAX_EPS) ? "" : "IN")
              << "CORRECT with accuracy: " << eps << std::endl;
  }
}

const unsigned max_size = 8192;
const unsigned min_size = 32;

void Usage(const char* name) {

  std::cout << " Calculating floating point matrix multiply on gpu\n";
  std::cout << name << " [ [gpu|cpu|host, default=gpu],  [matrix size, default=1024, max="
            << max_size << "], [repetition count, default=4]] \n";
}

inline static constexpr auto kNsInSec = 1'000'000'000;

struct BufferStore {
  public:
  inline static constexpr auto kRequestedBufferSize = 5'000'000 * sizeof(pti_view_record_kernel);
  inline static constexpr auto kRequestedAlignment = std::align_val_t{8};

  inline static auto& Instance() {
    static BufferStore buf_data{};
    return buf_data;
  }

  inline void RequestBuffer(unsigned char** buf_out, std::size_t* size_out) noexcept {
    try {
      buffer_store_.emplace_back(std::unique_ptr<unsigned char[]>(
            static_cast<unsigned char*>(
              ::operator new[](kRequestedBufferSize, kRequestedAlignment))));
      *buf_out = buffer_store_.back().get();
      *size_out = kRequestedBufferSize;
    } catch (const std::bad_alloc& e) {
      std::cerr << "Unable to allocate space for buffer. Aborting.." << '\n';
      std::abort();
    } catch (...) {
      std::cerr << "Unknown error caught. Aborting.." << '\n';
      std::abort();
    }
  }

  inline void ParseBuffer(unsigned char* buf, std::size_t buf_size, std::size_t valid_size) {
    try {
      if (!buf || !valid_size || !buf_size) {
        std::cerr << "Received empty buffer" << '\n';
        return;
      }
      pti_view_record_base *ptr = nullptr;
      while (true) {
        const auto buf_status =
            ptiViewGetNextRecord(buf, valid_size, &ptr);
        if (buf_status == pti_result::PTI_STATUS_END_OF_BUFFER) {
          break;
        }
        if (buf_status != pti_result::PTI_SUCCESS) {
          std::cerr << "Found Error Parsing Records from PTI" << '\n';
          break;
        }
        switch (ptr->_view_kind) {
          case pti_view_kind::PTI_VIEW_DEVICE_GPU_KERNEL: {
            const auto* record = reinterpret_cast<pti_view_record_kernel *>(ptr);
            const auto elapsed_time = static_cast<double>(record->_end_timestamp - record->_start_timestamp) / kNsInSec;
            kernel_record_store_.push_back(record);
          }
          default: {
            break;
          }
        }
      }
    } catch (...) {
      std::cerr << "Unknown error caught. Aborting.." << '\n';
      std::abort();
    }
  }

  inline auto& KernelRecords() {
    return kernel_record_store_;
  }

  private:
  BufferStore() {
  }
  std::vector<std::unique_ptr<unsigned char[]>> buffer_store_;
  std::vector<const pti_view_record_kernel *> kernel_record_store_;
};

inline void OnRequestBuffer(unsigned char** buf_out, std::size_t* size_out) {
  BufferStore::Instance().RequestBuffer(buf_out, size_out);
}

inline void OnReceivedBuffer(unsigned char* buf, std::size_t buf_size, std::size_t valid_size) {
  BufferStore::Instance().ParseBuffer(buf, buf_size, valid_size);
}


int main(int argc, char *argv[]) {
  int exit_code = EXIT_SUCCESS;
  unsigned repeat_count = 4;
  std::vector<double> kernel_time_storage{};
  kernel_time_storage.reserve(4);
  unsigned size = 1024;
  sycl::device dev;
  try {
    PTI_CHECK_RETURN(ptiViewSetCallbacks(OnRequestBuffer, OnReceivedBuffer));
    dev = sycl::device(sycl::gpu_selector_v);
    if (argc > 1 && strcmp(argv[1], "cpu") == 0) {
      dev = sycl::device(sycl::cpu_selector_v);
      std::cerr << "PTI doesn't support cpu profiling yet" << '\n';
      std::exit(EXIT_FAILURE);
    } else if (argc > 1 && strcmp(argv[1], "host") == 0) {
      dev = sycl::device(sycl::default_selector_v);
      std::cerr << "PTI doesn't support host profiling yet" << '\n';
      std::exit(EXIT_FAILURE);
    }

    unsigned temp = size;
    if (argc > 2) {
      temp = std::stoul(argv[2]);
      size = (temp < min_size) ? min_size :
                    (temp > max_size) ?  max_size : temp;
    }

    if (argc > 3) {
      temp = std::stoul(argv[3]);
      repeat_count = (temp < 1) ? 1 : temp;
    }
  } catch (const sycl::exception &e) {
    Usage(argv[0]);
    std::cerr << "Error: Exception caught while executing SYCL " << e.what() << '\n';
    std::cerr << "Unable to select valid sycl device" << '\n';
    return EXIT_FAILURE;
  } catch(...) {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }

  sycl::property_list prop_list{sycl::property::queue::enable_profiling()};
  sycl::queue queue(dev, sycl::async_handler{}, prop_list);

  std::cout << "DPC++ Matrix Multiplication (matrix size: " << size << " x "
            << size << ", repeats " << repeat_count << " times)" << std::endl;
  std::cout << "Target device: "
            << queue.get_info<sycl::info::queue::device>()
                  .get_info<sycl::info::device::name>()
            << std::endl;

  std::vector<float> a(size * size, A_VALUE);
  std::vector<float> b(size * size, B_VALUE);
  std::vector<float> c(size * size, 0.0f);

  try {
    auto start = std::chrono::steady_clock::now();
    float expected_result = A_VALUE * B_VALUE * size;
    Compute(queue, a, b, c, size, repeat_count, expected_result, nullptr);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> time = end - start;
    std::cout << "Total execution time without tracing: " << time.count() << " sec"
              << std::endl;


    StartTracing();
    start = std::chrono::steady_clock::now();
    expected_result = A_VALUE * B_VALUE * size;

    Compute(std::move(queue), a, b, c, size, repeat_count, expected_result, &kernel_time_storage);
    end = std::chrono::steady_clock::now();
    time = end - start;

    std::cout << "Total execution time with tracing: " << time.count()
              << " sec" << std::endl;
    StopTracing();
    PTI_CHECK_RETURN(ptiFlushAllViews());
  } catch (const sycl::exception &e) {
    std::cerr << "Error: Exception while executing SYCL " << e.what() << '\n';
    std::cerr << "\tError code: " << e.code().value()
              << "\n\tCategory: " << e.category().name()
              << "\n\tMessage: " << e.code().message() << '\n';
    exit_code = EXIT_FAILURE;
  } catch (const std::exception &e) {
    std::cerr << "Error: Exception caught " << e.what() << '\n';
    exit_code = EXIT_FAILURE;
  } catch (...) {
    std::cerr << "Error: Unknown exception caught." << '\n';
    exit_code = EXIT_FAILURE;
  }

  PTI_CHECK_RETURN(ptiFlushAllViews());
  std::cout << "Number of Kernel Records "   << BufferStore::Instance().KernelRecords().size() << '\n';
  std::cout << "Number of Kernel Records (SYCL) "   <<  kernel_time_storage.size() << '\n';

  std::sort(std::begin(BufferStore::Instance().KernelRecords()),
      std::end(BufferStore::Instance().KernelRecords()),
      [](const auto* k_record_1, const auto* k_record_2){
        return k_record_1->_end_timestamp < k_record_2->_end_timestamp;
      });

  for(std::size_t k_idx = 0; k_idx < BufferStore::Instance().KernelRecords().size(); ++k_idx) {
    const auto* record = BufferStore::Instance().KernelRecords()[k_idx];
    const auto elapsed_time = static_cast<double>(record->_end_timestamp - record->_start_timestamp) / kNsInSec;
    std::cout << "First Kernel Elapsed Time According to PTI " << elapsed_time << '\n';
    std::cout << "First Kernel Elapsed Time According to SYCL " << kernel_time_storage[k_idx] << '\n';
    std::cout << "Are equal? " << (elapsed_time == kernel_time_storage[k_idx]) << '\n';
  }


  return exit_code;
}
