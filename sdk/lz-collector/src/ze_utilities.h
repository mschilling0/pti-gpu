//============================================================== Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef ZE_UTILITIES_H_
#define ZE_UTILITIES_H_

#include <level_zero/layers/zel_tracing_api.h>
#include <level_zero/ze_api.h>

#include <cstddef>
#include <exception>
#include <iostream>
#include <ostream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pti {
namespace utils {
namespace ze {
constexpr inline auto kNSecInSecond = 1'000'000'000ULL;

constexpr inline uint64_t GetMask(uint32_t valid_bits) {
  return static_cast<uint64_t>((1ULL << valid_bits) - 1ULL);
}

// L0 gives us the cycle count so we need the # ns in a cycle
// (nsec/sec)/(cycles/sec) = nsec/cycle
constexpr inline uint64_t NSecPerCycle(uint64_t timer_resolution) {
  return kNSecInSecond / timer_resolution;
}

std::ostream& operator<<(std::ostream& out, ze_result_t ze_result) {
  switch (ze_result) {
    case ze_result_t::ZE_RESULT_SUCCESS:
      out << "ZE_RESULT_SUCCES";
      break;
    case ze_result_t::ZE_RESULT_NOT_READY:
      out << "ZE_RESULT_NOT_READY";
      break;
    case ze_result_t::ZE_RESULT_ERROR_DEVICE_LOST:
      out << "ZE_RESULT_ERROR_DEVICE_LOST";
      break;
    case ze_result_t::ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY:
      out << "ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY";
      break;
    case ze_result_t::ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY:
      out << "ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY";
      break;
    case ze_result_t::ZE_RESULT_WARNING_DROPPED_DATA:
      out << "ZE_RESULT_WARNING_DROPPED_DATA";
      break;
    case ze_result_t::ZE_RESULT_ERROR_UNINITIALIZED:
      out << "ZE_RESULT_ERROR_UNINITIALIZED";
      break;
    case ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_VERSION:
      out << "ZE_RESULT_ERROR_UNSUPPORTED_VERSION";
      break;
    case ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_FEATURE:
      out << "ZE_RESULT_ERROR_UNSUPPORTED_FEATURE";
      break;
    case ze_result_t::ZE_RESULT_ERROR_INVALID_ARGUMENT:
      out << "ZE_RESULT_ERROR_INVALID_ARGUMENT";
      break;
    case ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_HANDLE:
      out << "ZE_RESULT_ERROR_INVALID_NULL_HANDLE";
      break;
    case ze_result_t::ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE:
      out << "ZE_RESULT_ERROR_HANDLE_OBJECT_IN_USE";
      break;
    case ze_result_t::ZE_RESULT_ERROR_INVALID_NULL_POINTER:
      out << "ZE_RESULT_ERROR_INVALID_NULL_POINTER";
      break;
    case ze_result_t::ZE_RESULT_ERROR_INVALID_SIZE:
      out << "ZE_RESULT_ERROR_INVALID_SIZE";
      break;
    case ze_result_t::ZE_RESULT_ERROR_UNSUPPORTED_SIZE:
      out << "ZE_RESULT_ERROR_UNSUPPORTED_SIZE";
      break;
    default:
      out << "<unknown ze::Err: " << static_cast<std::size_t>(ze_result) << ">";
      break;
  }
  return out;
}

class Err : public std::exception {
 public:
  explicit Err(std::string_view err_msg, ze_result_t res) : what_msg_(err_msg), res_(res) {}

  [[nodiscard]] inline const char* what() const noexcept override { return std::data(what_msg_); }

  [[nodiscard]] inline ze_result_t ReturnValue() const noexcept { return res_; }

 private:
  std::string_view what_msg_;
  ze_result_t res_;
};

inline std::string GetKernelName(ze_kernel_handle_t kernel) {
  std::size_t size = 0;
  ze_result_t status = zeKernelGetName(kernel, &size, nullptr);
  if (status != ze_result_t::ZE_RESULT_SUCCESS || !size) {
    std::cerr << "Unable to get kernel size: " << status << '\n';
    return std::string{};
  }

  std::vector<char> name(size);
  status = zeKernelGetName(kernel, &size, name.data());

  if (status != ze_result_t::ZE_RESULT_SUCCESS) {
    std::cerr << "Unable to get kernel name: " << status << '\n';
    return std::string{};
  }

  return std::string(name.begin(), name.end() - 1);
}

class Tracer {
 public:
  explicit Tracer(const zel_tracer_desc_t& descriptor) {
    auto status = zelTracerCreate(&descriptor, &tracer_);
    if (status != ze_result_t::ZE_RESULT_SUCCESS) {
      throw Err("Unable to create tracer", status);
    }
  }
  Tracer(const Tracer&) = delete;
  Tracer(Tracer&&) = delete;
  Tracer& operator=(const Tracer&) = delete;
  Tracer& operator=(Tracer&&) = delete;

  void Disable() {
    tracer_enabled_ = 0;
    auto status = zelTracerSetEnabled(tracer_, tracer_enabled_);
    if (status != ze_result_t::ZE_RESULT_SUCCESS) {
      throw Err("Unable to disable tracer", status);
    }
  }

  void Enable() {
    tracer_enabled_ = 1;
    auto status = zelTracerSetEnabled(tracer_, tracer_enabled_);
    if (status != ze_result_t::ZE_RESULT_SUCCESS) {
      throw Err("Unable to enable tracer", status);
    }
  }

  void SetPrologues(zel_core_callbacks_t& callbacks) {
    auto status = zelTracerSetPrologues(tracer_, &callbacks);
    if (status != ze_result_t::ZE_RESULT_SUCCESS) {
      throw Err("Unable to set prologues", status);
    }
  }

  void SetEpilogues(zel_core_callbacks_t& callbacks) {
    auto status = zelTracerSetEpilogues(tracer_, &callbacks);
    if (status != ze_result_t::ZE_RESULT_SUCCESS) {
      throw Err("Unable to set epiologues", status);
    }
  }

  virtual ~Tracer() {
    if (tracer_) {
      ze_result_t status = zelTracerDestroy(tracer_);
      if (status != ze_result_t::ZE_RESULT_SUCCESS) {
        std::cerr << "Unable to destroy tracer " << status << '\n';
      }
    }
  }

 private:
  zel_tracer_handle_t tracer_ = nullptr;
  ze_bool_t tracer_enabled_ = 0;
};

enum class ZeMemoryType { kHost, kShared, kDevice };

template <ZeMemoryType MemT>
class ZeMemory {
 public:
  template <typename... T>
  explicit ZeMemory(ze_context_handle_t ctx, T... args) : ctx_(ctx) {
    ze_result_t result = ze_result_t::ZE_RESULT_SUCCESS;
    if constexpr (MemT == ZeMemoryType::kDevice) {
      result = zeMemAllocDevice(ctx_, args..., &mem_);
    } else if constexpr (MemT == ZeMemoryType::kShared) {
      result = zeMemAllocShared(ctx_, args..., &mem_);
    } else {
      result = zeMemAllocHost(ctx_, args..., &mem_);
    }
    if (result != ze_result_t::ZE_RESULT_SUCCESS) {
      throw Err("Unable to allocate memory", result);
    }
  }

  constexpr void* Get() { return mem_; }

  ZeMemory(const ZeMemory&) = delete;
  ZeMemory& operator=(const ZeMemory&) = delete;

  ZeMemory(ZeMemory&& other)
      : mem_(std::exchange(other.mem_, nullptr)), ctx_(std::exchange(other.ctx_, nullptr)) {}
  ZeMemory& operator=(ZeMemory&& other) {
    if (this != &other) {
      mem_ = std::exchange(other.mem_, nullptr);
      ctx_ = std::exchange(other.ctx_, nullptr);
    }
    return *this;
  }

  virtual ~ZeMemory() {
    if(mem_) {
      auto result = zeMemFree(ctx_, mem_);
      if (result != ze_result_t::ZE_RESULT_SUCCESS) {
        std::cerr << "Error freeing memory...." << '\n';
      }
    }
  }

 private:
  void* mem_ = nullptr;
  ze_context_handle_t ctx_ = nullptr;
};

}  // namespace ze
}  // namespace utils
}  // namespace pti

#endif  // ZE_UTILITIES_H_
