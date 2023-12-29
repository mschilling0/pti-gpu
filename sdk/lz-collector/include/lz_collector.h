//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================
#ifndef LZ_COLLECTOR_H_
#define LZ_COLLECTOR_H_

class LzCollector {
public:
  LzCollector();
  LzCollector(const LzCollector&) = delete;
  LzCollector(LzCollector&&) = delete;
  LzCollector& operator=(const LzCollector&) = delete;
  LzCollector& operator=(LzCollector&&) = delete;
  virtual ~LzCollector();

  void PrintResults();

private:
  void* handle_ = nullptr;
};

#endif  // LZ_COLLECTOR_H_
