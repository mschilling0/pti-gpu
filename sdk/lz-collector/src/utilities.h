//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef UTILITIES_H_
#define UTILITIES_H_

#include <level_zero/layers/zel_tracing_api.h>
#include <level_zero/ze_api.h>

#include <exception>
#include <string_view>
#include <system_error>

namespace pti {
namespace utils {

template <typename T>
constexpr inline void CheckPosixRet(T ret_value) {
  static_assert(std::is_signed<T>::value);
  if (ret_value < 0) {
    throw std::system_error{ret_value, std::system_category()};
  }
}

inline void SetEnv(const char* name, const char* value) {
  auto status = setenv(name, value, 1);
  CheckPosixRet(status);
}
}  // namespace utils
}  // namespace pti

#endif  // UTILITIES_H_
