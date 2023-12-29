#include <sycl/sycl.hpp>
#include "lz_collector.h"

namespace syclex = sycl::ext::oneapi::experimental;

int main() {
  for (const auto &dev : sycl::device::get_devices()) {
    using graph_support = syclex::info::device::graph_support;
    using gsl = syclex::info::graph_support_level;
    const auto gs = dev.get_info<graph_support>();
    std::cout << dev.get_info<sycl::info::device::name>() << " : "
              << (gs == gsl::unsupported
                      ? "unsupported"
                      : (gs == gsl::emulated ? "emulated" : "native"))
              << std::endl;
    LzCollector collector{};
    if (gs != gsl::unsupported) {
      sycl::context ctx{dev};
      sycl::queue q1{ctx, dev, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling(), sycl::ext::intel::property::queue::no_immediate_command_list()}};
      std::vector<sycl::queue> queuesToRecord{q1};

      const sycl::property_list propList{syclex::property::graph::no_cycle_check()};
      syclex::command_graph<syclex::graph_state::modifiable> graph(ctx, dev, propList);

      int *value_h = sycl::malloc_host<int>(1, ctx);
      int *value_i = sycl::malloc_device<int>(1, dev, ctx);
      int *value_o = sycl::malloc_device<int>(1, dev, ctx);

      value_h[0] = 1;

      q1.memcpy(value_i, value_h, 1 * sizeof(int)).wait_and_throw();

      bool result = graph.begin_recording(queuesToRecord);
      if (!result) {
        std::cout << "  Could not start the recording" << std::endl;
      }

      q1.submit([&](sycl::handler &cgh) {
        cgh.single_task<class Memset>([=]() { value_o[0] = 0; });
      });
      q1.submit([&](sycl::handler &cgh) {
        cgh.single_task<class Memcpy>([=]() { value_i[0] = value_o[0]; });
      });

      graph.end_recording();
      auto instance = graph.finalize();

      q1.ext_oneapi_graph(instance).wait_and_throw();
      std::cout << "   Done!" << std::endl;
      q1.wait_and_throw();
      collector.PrintResults();
    } // Here it dies when destroying `instance`
  }
  std::cout << "Done!" << std::endl;
  return 0;
}
