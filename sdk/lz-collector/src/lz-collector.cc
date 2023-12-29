#include <stdexcept>

#include "lz_collector.h"
#include "ze_kernel_collector.h"

namespace {
class GlobalZeInitializer {
 public:
  inline static ze_result_t Initialize() {
    utils::SetEnv("ZE_ENABLE_TRACING_LAYER", "1");
    ze_result_t status = zeInit(ZE_INIT_FLAG_GPU_ONLY);
    return status;
  }

  inline static ze_result_t result_ = Initialize();
};

}  // namespace

void LzCollector::PrintResults() {
  auto* collector = static_cast<ZeKernelCollector*>(handle_);

  if (!collector) {
    throw std::runtime_error("UNable to print results");
  }

  const ZeKernelInfoMap& kernel_info_map = collector->GetKernelInfoMap();
  if (kernel_info_map.size() == 0) {
    return;
  }

  uint64_t total_duration = 0;
  for (auto& value : kernel_info_map) {
    total_duration += value.second.total_time;
  }

  std::cerr << std::endl;
  std::cerr << "=== Device Timing Results: ===" << std::endl;
  std::cerr << std::endl;
  std::cerr << "Total Device Time (ns): " << total_duration << std::endl;
  std::cerr << std::endl;

  if (total_duration > 0) {
    ZeKernelCollector::PrintKernelsTable(kernel_info_map);
  }

  std::cerr << std::endl;
}

LzCollector::LzCollector() {
  if (GlobalZeInitializer::result_ != ZE_RESULT_SUCCESS) {
    throw std::runtime_error("zeInit failed");
  }

  ZeKernelCollector* collector = ZeKernelCollector::Create();

  if (!collector) {
    delete collector;
    throw std::runtime_error("Create collector failed..");
  }

  handle_ = static_cast<void*>(collector);
}

LzCollector::~LzCollector() {
  auto* collector = static_cast<ZeKernelCollector*>(handle_);

  if (collector) {
    collector->DisableTracing();
  }

  delete collector;
}
