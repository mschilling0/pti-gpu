#ifndef PTI_ADAPTER_H_
#define PTI_ADAPTER_H_

#include <string_view>

extern "C" {
  __attribute__ ((visibility ("default"))) void PtiAdapterInit();
}

struct Env {
  inline static std::string_view kActivatePti = "PTI_COLLECTOR_ENABLED";
};


#endif // PTI_ADAPTER_H_
