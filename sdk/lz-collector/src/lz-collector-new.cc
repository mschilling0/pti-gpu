#include <level_zero/ze_api.h>

#include <iostream>
#include <memory>
#include <system_error>

#include "lz_collector.h"
#include "utilities.h"
#include "ze_kernel_view_collector.h"

namespace {
class GlobalZeInitializer {
 public:
  inline static ze_result_t Initialize() {
    try {
      pti::utils::SetEnv("ZE_ENABLE_TRACING_LAYER", "1");
    } catch (const std::system_error& e) {
      std::cerr << "Unable to enable ze tracing layer: " << e.code().message() << '\n';
      std::terminate();
    }
    return zeInit(ZE_INIT_FLAG_GPU_ONLY);
  }
  inline static ze_result_t result_ = Initialize();
};

}  // namespace

void LzCollector::PrintResults() {}

LzCollector::LzCollector() : handle_(new pti::ZeKernelViewCollector{}) {
  static_cast<pti::ZeKernelViewCollector*>(handle_)->Start();
}

LzCollector::~LzCollector() { delete static_cast<pti::ZeKernelViewCollector*>(handle_); }
