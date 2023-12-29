//==============================================================
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

#ifndef ZE_KERNEL_VIEW_COLLECTOR_H_
#define ZE_KERNEL_VIEW_COLLECTOR_H_

#include <level_zero/layers/zel_tracing_api.h>
#include <level_zero/ze_api.h>

#include <cstdint>
#include <iostream>
#include <unordered_map>

#include "view_buffer.h"
#include "ze_utilities.h"

namespace pti {

namespace view {
/**
 * @brief Kind of software and hardware operations to be tracked and viewed,
 * passed to ptiViewEnable/ptiViewDisable
 */
typedef enum _pti_view_kind {
  PTI_VIEW_INVALID = 0,
  PTI_VIEW_DEVICE_GPU_KERNEL = 1,     //!< Device kernels
  PTI_VIEW_DEVICE_CPU_KERNEL = 2,     //!< Host (CPU) kernels
  PTI_VIEW_LEVEL_ZERO_CALLS = 3,      //!< Level-Zero APIs tracing
  PTI_VIEW_OPENCL_CALLS = 4,          //!< OpenCL APIs tracing
  PTI_VIEW_COLLECTION_OVERHEAD = 5,   //!< Collection overhead
  PTI_VIEW_SYCL_RUNTIME_CALLS = 6,    //!< SYCL runtime API tracing
  PTI_VIEW_EXTERNAL_CORRELATION = 7,  //!< Correlation of external operations
  PTI_VIEW_DEVICE_GPU_MEM_COPY = 8,   //!< Memory copies between Host and Device
  PTI_VIEW_DEVICE_GPU_MEM_FILL = 9,   //!< Device memory fills
} pti_view_kind;

/**
 * @brief Base View record type
 */
typedef struct pti_view_record_base {
  pti_view_kind _view_kind;  //!< Record View kind
} pti_view_record_base;

typedef struct pti_view_record_kernel {
  pti_view_record_base _view_kind;          //!< Base record
  ze_command_queue_handle_t _queue_handle;  //!< Device back-end queue handle
  ze_device_handle_t _device_handle;        //!< Device handle
  ze_context_handle_t _context_handle;      //!< Context handle
  const char* _name;                        //!< Kernel name
  const char* _source_file_name;            //!< Kernel source file,
                                            //!< null if no information
  uint64_t _source_line_number;             //!< Kernel beginning source line number,
                                            //!< 0 if no information
  uint64_t _kernel_id;                      //!< Kernel instance ID,
                                            //!< unique among all device kernel instances
  uint32_t _correlation_id;                 //!< ID that correlates this record with records
                                            //!< of other Views
  uint32_t _thread_id;                      //!< Thread ID of Function call
  char _pci_address[16];                    //!< Device pci_address
  uint64_t _append_timestamp;               //!< Timestamp of kernel appending to
                                            //!< back-end command list, ns
  uint64_t _start_timestamp;                //!< Timestamp of kernel start on device, ns
  uint64_t _end_timestamp;                  //!< Timestamp of kernel completion on device, ns
  uint64_t _submit_timestamp;               //!< Timestamp of kernel command list submission
                                            //!< of device, ns
  uint64_t _sycl_task_begin_timestamp;      //!< Timestamp of kernel submission from SYCL layer,
                                            //!< ns
  uint64_t _sycl_enqk_begin_timestamp;
  uint64_t _sycl_node_id;
  uint32_t _sycl_invocation_id;
} pti_view_record_kernel;

}  // namespace view

std::ostream& operator<<(std::ostream& out, view::pti_view_record_kernel record) {
  if (record._name) {
    out << record._name;
  }
  return out;
}

struct CommandListMetaData {
  ze_context_handle_t ctx = nullptr;
  ze_device_handle_t dev = nullptr;
  bool immediate = false;
};

void CreateEvent(ze_context_handle_t context, ze_event_pool_handle_t& event_pool,
                 ze_event_handle_t& event) {
  ze_result_t status = ZE_RESULT_SUCCESS;

  ze_event_pool_desc_t event_pool_desc = {
      ZE_STRUCTURE_TYPE_EVENT_POOL_DESC, nullptr,
      ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP | ZE_EVENT_POOL_FLAG_HOST_VISIBLE, 1};
  status = zeEventPoolCreate(context, &event_pool_desc, 0, nullptr, &event_pool);

  if (status != ZE_RESULT_SUCCESS) {
    throw utils::ze::Err("Unable to create event pool", status);
  }

  ze_event_desc_t event_desc = {ZE_STRUCTURE_TYPE_EVENT_DESC, nullptr, 0,  ZE_EVENT_SCOPE_FLAG_HOST,//ZE_EVENT_SCOPE_FLAG_HOST,
                                ZE_EVENT_SCOPE_FLAG_HOST};
  status = zeEventCreate(event_pool, &event_desc, &event);
  if (status != ZE_RESULT_SUCCESS) {
    throw utils::ze::Err("Unable to create event pool", status);
  }
}

constexpr std::size_t kDefaultBufferSize = 1'000'000;

void OnEnterEventDestroy(ze_event_destroy_params_t* params, ze_result_t result, void* global_data,
                         void** instance_data);
//void OnEnterEventCreate(ze_event_create_params_t* params, ze_result_t result, void* global_data,
//                         void** instance_data);
void OnEnterEventHostReset(ze_event_host_reset_params_t* params, ze_result_t result,
                           void* global_data, void** instance_data);
void OnEnterEventPoolCreate(ze_event_pool_create_params_t* params, ze_result_t result,
                            void* global_data, void** instance_data);
void OnExitEventPoolCreate(ze_event_pool_create_params_t* params, ze_result_t result,
                           void* global_data, void** instance_data);
void OnExitEventHostSynchronize(ze_event_host_synchronize_params_t* params, ze_result_t result,
                                void* global_data, void** instance_data);
void OnEnterCommandListAppendLaunchKernel(ze_command_list_append_launch_kernel_params_t* params,
                                          ze_result_t result, void* global_data,
                                          void** instance_data);
void OnExitCommandListAppendLaunchKernel(ze_command_list_append_launch_kernel_params_t* params,
                                         ze_result_t result, void* global_data,
                                         void** instance_data);
void OnExitCommandListCreate(ze_command_list_create_params_t* params, ze_result_t result,
                             void* global_data, void** instance_data);
void OnExitCommandListCreateImmediate(ze_command_list_create_immediate_params_t* params,
                                      ze_result_t result, void* global_data, void** instance_data);
void OnExitCommandListDestroy(ze_command_list_destroy_params_t* params, ze_result_t result,
                              void* global_data, void** instance_data);
void OnExitCommandListReset(ze_command_list_reset_params_t* params, ze_result_t result,
                            void* global_data, void** instance_data);
void OnExitCommandQueueSynchronize(ze_command_queue_synchronize_params_t* params,
                                   ze_result_t result, void* global_data, void** instance_data);
void OnExitCommandQueueDestroy(ze_command_queue_destroy_params_t* params, ze_result_t result,
                               void* global_data, void** instance_data);
void OnExitCommandQueueExecuteCommandLists(ze_command_queue_execute_command_lists_params_t* params,
                                           ze_result_t result, void* global_data,
                                           void** instance_data);

struct TimestampListEntry {
  ze_event_handle_t event = nullptr;
  utils::ze::ZeMemory<utils::ze::ZeMemoryType::kShared> mem;
};

class ZeKernelViewCollector {
 public:
  ZeKernelViewCollector()
      : tracer_{{ZEL_STRUCTURE_TYPE_TRACER_DESC, nullptr, this}},
        buffer_memory_(kDefaultBufferSize) {
    SetTracingCallbacks();
    buffer_.Refresh(buffer_memory_.data(), buffer_memory_.size());
    command_list_map_.reserve(1);
  }

  virtual ~ZeKernelViewCollector() {
    Stop();
    std::cout << "Printing Kernels..." << '\n';
    std::size_t accum = 0;
    unsigned char* current_record = buffer_.GetBuffer();
    while (current_record != buffer_.GetRecordsEnd()) {
      auto* formed_record = reinterpret_cast<view::pti_view_record_kernel*>(current_record);
      if (formed_record->_view_kind._view_kind == view::PTI_VIEW_DEVICE_GPU_KERNEL) {
        std::cout << "Found Kernel: " << *formed_record << '\n';
        accum += 1;
      }
      current_record += sizeof(view::pti_view_record_kernel);
    }
    std::cout << "Found " << accum << " Kernels." << '\n';
  }

  constexpr void Start() { tracer_.Enable(); }

  constexpr void Stop() { tracer_.Disable(); }

  constexpr view::utilities::ViewBuffer& GetBuffer() { return buffer_; }

  constexpr std::vector<std::string>& GetKernelNames() { return kernel_names_; }

  constexpr auto& GetEventList() { return ts_list_; }

  inline void StoreCommandListMetadata(ze_command_list_handle_t list, ze_context_handle_t ctx,
                                       ze_device_handle_t dev, bool immediate) {
    command_list_map_.emplace(list, CommandListMetaData{ctx, dev, immediate});
  }

  inline CommandListMetaData GetCommandListMetadata(ze_command_list_handle_t list) const {
    const auto command_list_info = command_list_map_.find(list);

    if (command_list_info == std::end(command_list_map_)) {
      return CommandListMetaData{};
    }

    return command_list_info->second;
  }

  static ze_event_pool_desc_t* GetEventPoolDesc(const ze_event_pool_desc_t* event_pool_desc) {
    event_pool_desc_.stype = event_pool_desc->stype;
    event_pool_desc_.pNext = event_pool_desc->pNext;
    event_pool_desc_.flags = event_pool_desc->flags;
    event_pool_desc_.flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    event_pool_desc_.flags |= ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    return &event_pool_desc_;
  }

 private:
  void SetTracingCallbacks() {
    zel_core_callbacks_t prologue_callbacks{};
    zel_core_callbacks_t epilogue_callbacks{};

    //prologue_callbacks.Event.pfnCreateCb = OnEnterEventCreate;
    // zeEventDestroy
    prologue_callbacks.Event.pfnDestroyCb = OnEnterEventDestroy;
    // zeEventHostReset
    prologue_callbacks.Event.pfnHostResetCb = OnEnterEventHostReset;
    // zeEventPoolCreate
    prologue_callbacks.EventPool.pfnCreateCb = OnEnterEventPoolCreate;
    epilogue_callbacks.EventPool.pfnCreateCb = OnExitEventPoolCreate;
    // zeEventHostSynchronize
    epilogue_callbacks.Event.pfnHostSynchronizeCb = OnExitEventHostSynchronize;
    // zeCommandListAppendLaunchKernel
    prologue_callbacks.CommandList.pfnAppendLaunchKernelCb = OnEnterCommandListAppendLaunchKernel;
    epilogue_callbacks.CommandList.pfnAppendLaunchKernelCb = OnExitCommandListAppendLaunchKernel;
    // zeCommandListCreate
    epilogue_callbacks.CommandList.pfnCreateCb = OnExitCommandListCreate;
    // zeCommandListCreateImmediate
    epilogue_callbacks.CommandList.pfnCreateImmediateCb = OnExitCommandListCreateImmediate;
    // zeCommandListDestroy
    epilogue_callbacks.CommandList.pfnDestroyCb = OnExitCommandListDestroy;
    // zeCommandListReset
    epilogue_callbacks.CommandList.pfnResetCb = OnExitCommandListReset;
    // zeCommandQueueExecuteCommandLists
    epilogue_callbacks.CommandQueue.pfnExecuteCommandListsCb =
        OnExitCommandQueueExecuteCommandLists;
    // zeCommandQueueSychronize
    epilogue_callbacks.CommandQueue.pfnSynchronizeCb = OnExitCommandQueueSynchronize;
    // zeCommandQueueDestroy
    epilogue_callbacks.CommandQueue.pfnDestroyCb = OnExitCommandQueueDestroy;

    tracer_.SetPrologues(prologue_callbacks);
    tracer_.SetEpilogues(epilogue_callbacks);
  }

  utils::ze::Tracer tracer_;
  std::vector<unsigned char> buffer_memory_;
  view::utilities::ViewBuffer buffer_;
  std::vector<std::string> kernel_names_;
  inline static thread_local ze_event_pool_desc_t event_pool_desc_;
  std::unordered_map<ze_command_list_handle_t, CommandListMetaData> command_list_map_ = {};
  //std::vector<std::future<bool>> ts_event_list_ = {};
  std::vector<TimestampListEntry> ts_list_ = {};
};

// zeEventPoolCreate
void OnEnterEventPoolCreate(ze_event_pool_create_params_t* params, ze_result_t result,
                            void* global_data, void** instance_data) {
  if (!params || !params->pdesc) {
    return;
  }

  const auto* pool_desc = *(params->pdesc);
  if (!pool_desc || pool_desc->flags & ZE_EVENT_POOL_FLAG_IPC) {
    return;
  }

  auto* collector = static_cast<ZeKernelViewCollector*>(global_data);
  if (!collector) {
    return;
  }

  auto* new_pool_desc =  new ze_event_pool_desc_t;// ZeKernelViewCollector::GetEventPoolDesc(pool_desc);
    new_pool_desc->stype = pool_desc->stype;
    // PTI_ASSERT(new_pool_desc->stype == ZE_STRUCTURE_TYPE_EVENT_POOL_DESC);
    new_pool_desc->pNext = pool_desc->pNext;
    new_pool_desc->flags = pool_desc->flags;
    new_pool_desc->flags |= ZE_EVENT_POOL_FLAG_KERNEL_TIMESTAMP;
    new_pool_desc->flags |= ZE_EVENT_POOL_FLAG_HOST_VISIBLE;
    new_pool_desc->count = pool_desc->count;

    *(params->pdesc) = new_pool_desc;
    *instance_data = new_pool_desc;


  std::cout << __PRETTY_FUNCTION__ << '\n';
}

void OnExitEventPoolCreate(ze_event_pool_create_params_t* params, ze_result_t result,
                           void* global_data, void** instance_data) {
  std::cout << __PRETTY_FUNCTION__ << '\n';
  ze_event_pool_desc_t* desc = static_cast<ze_event_pool_desc_t*>(*instance_data);
  delete desc;
}

// zeEventDestroy
void OnEnterEventDestroy(ze_event_destroy_params_t* params, ze_result_t result, void* global_data,
                         void** instance_data) {
  std::cout << __PRETTY_FUNCTION__ << '\n';
}

// zeEventHostReset
void OnEnterEventHostReset(ze_event_host_reset_params_t* params, ze_result_t result,
                           void* global_data, void** instance_data) {
  std::cout << __PRETTY_FUNCTION__ << '\n';
}

// zeEventHostSynchronize
void OnExitEventHostSynchronize(ze_event_host_synchronize_params_t* params, ze_result_t result,
                                void* global_data, void** instance_data) {
  std::cout << __PRETTY_FUNCTION__ << '\n';
}

// zeCommandListAppendLaunchKernel
void OnEnterCommandListAppendLaunchKernel(ze_command_list_append_launch_kernel_params_t* params,
                                          ze_result_t result, void* global_data,
                                          void** instance_data) {
  std::cout << __PRETTY_FUNCTION__ << '\n';
  auto* collector = static_cast<ZeKernelViewCollector*>(global_data);
  if (!collector) {
    return;
  }

  auto& buffer = collector->GetBuffer();

  if (buffer.IsNull()) {
    std::cerr << "Null buf cannot insert... (TODO:REMOVE)" << '\n';
    return;
  }

  // TODO(matthew.schilling@intel.com): Record Creation Template
  // Create Kernel Record
  view::pti_view_record_kernel kernel;
  std::memset(&kernel, 0, sizeof(kernel));
  kernel._view_kind._view_kind = view::PTI_VIEW_DEVICE_GPU_KERNEL;
  /////////////////////////////////////////////////////////////////

  //  TODO(matthew.schilling@intel.com): Add to string table
  kernel._name = collector->GetKernelNames()
                     .emplace_back(std::move(utils::ze::GetKernelName(*(params->phKernel))))
                     .c_str();
  /////////////////////////////////////////////////////////////////
  try {
    if (!*(params->phSignalEvent)) {
      auto metadata = collector->GetCommandListMetadata(*(params->phCommandList));
      ze_event_pool_handle_t event_pool = nullptr;
      ze_event_handle_t event = nullptr;
      std::cout << "Creating event because no signal event specified" <<'\n';
      CreateEvent(metadata.ctx, event_pool, event);
      std::cout << "End Creating event because no signal event specified" <<'\n';
      *(params->phSignalEvent) = event;
    }
  } catch (const utils::ze::Err& e) {
    std::cerr << "Unable to create event " << e.what() << '\n';
    std::cerr << "Timestamp data will not be captured" << '\n';
  }

  auto* record = buffer.Insert(kernel);

  *instance_data = static_cast<void*>(record);
}

void OnExitCommandListAppendLaunchKernel(ze_command_list_append_launch_kernel_params_t* params,
                                         ze_result_t result, void* global_data,
                                         void** instance_data) {
  if (result == ze_result_t::ZE_RESULT_SUCCESS) {
    using utils::ze::ZeMemory;
    using utils::ze::ZeMemoryType;
    auto* collector = static_cast<ZeKernelViewCollector*>(global_data);
    if (!collector) {
      return;
    }
    auto* ze_cl_list = *(params->phCommandList);
    auto ze_cl_metadata = collector->GetCommandListMetadata(ze_cl_list);
    if(!ze_cl_metadata.dev) {
      std::cerr << "Device information not traced, cannot get kernel timestamps" << '\n';
      return;
    }
    ze_device_mem_alloc_desc_t ts_result_desc = {ZE_STRUCTURE_TYPE_DEVICE_MEM_ALLOC_DESC, nullptr,
                                                 0, 0};
    ze_host_mem_alloc_desc_t ts_result_desc2 = {ZE_STRUCTURE_TYPE_HOST_MEM_ALLOC_DESC, nullptr,
                                                 0};
    ZeMemory<ZeMemoryType::kShared> mem(ze_cl_metadata.ctx, &ts_result_desc, &ts_result_desc2,
                                        sizeof(ze_kernel_timestamp_result_t), alignof(uint32_t),
                                        ze_cl_metadata.dev);
    auto* ts_result = static_cast<ze_kernel_timestamp_result_t*>(mem.Get());
    ze_event_pool_handle_t event_pool = nullptr;
    ze_event_handle_t event = nullptr;
    CreateEvent(ze_cl_metadata.ctx, event_pool, event);
    auto result =
        zeCommandListAppendQueryKernelTimestamps(ze_cl_list, 1U, params->phSignalEvent, ts_result,
                                                 nullptr, event, 1U, params->phSignalEvent);
    if(!event || result != ZE_RESULT_SUCCESS) {
      std::cerr << "unable to append timestamp query " << result << '\n';
    } else {
      std::cout << "Appended timestamp query" << '\n';
    }
    ze_device_properties_t device_properties = {};
    device_properties.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
    zeDeviceGetProperties(ze_cl_metadata.dev, &device_properties);
    collector->GetEventList().push_back({event, std::move(mem)});
    //collector->GetEventList().emplace_back(std::async([mem = std::move(mem), event, ze_cl_list, &ze_cl_metadata,&device_properties, event2 = params->phSignalEvent]() mutable {
    //  auto result = zeEventHostSynchronize(event, UINT64_MAX);
    // if (result == ZE_RESULT_SUCCESS) {
    //   std::cout << "Found kernel timestamp!" << '\n';
    //    auto* ts_result = static_cast<ze_kernel_timestamp_result_t*>(mem.Get());
    //    if (ts_result) {
    //      ze_kernel_timestamp_result_t hs_res; // = *ts_result;
    //      //result = zeEventQueryKernelTimestamp(*event2, &hs_res);
    //      //std::memset(&hs_res, 0, sizeof(hs_res));
   //       zeCommandListAppendMemoryCopy(ze_cl_list, &hs_res, ts_result, sizeof(hs_res), nullptr, 0, nullptr);
   //       //std::cout <<  "\tKernel Start: " << (hs_res.global.kernelStart) << '\n';
   //       //std::cout <<  "\tKernel End: " <<  (hs_res.global.kernelEnd) <<  '\n';
   //       std::cout <<  "\tKernel Duration: " <<(((hs_res.global.kernelEnd - hs_res.global.kernelStart) & utils::ze::GetMask(device_properties.kernelTimestampValidBits))* utils::ze::NsPerCycle(device_properties.timerResolution)) << '\n'; // * utils::ze::NsPerCycle(device_properties.timerResolution)) <<  '\n';
   //     }
   //   }
   //   else {
   //     std::cerr << "Err: "  << result <<'\n';
   //   }
    //  result = zeEventDestroy(event);
    //  if (result != ZE_RESULT_SUCCESS) {
    //    std::cout << "Failed to destroy event" << '\n';
    // }
    //  return true;
    //}));
  }
}

// zeCommandListCreate
void OnExitCommandListCreate(ze_command_list_create_params_t* params, ze_result_t result,
                             void* global_data, void** instance_data) {
  std::cout << __PRETTY_FUNCTION__ << '\n';
  if (result == ze_result_t::ZE_RESULT_SUCCESS) {
    auto* collector = static_cast<ZeKernelViewCollector*>(global_data);
    if (!collector) {
      return;
    }
    collector->StoreCommandListMetadata(**(params->pphCommandList), *(params->phContext),
                                        *(params->phDevice), false);
  }
}

// zeCommandListCreateImmediate
void OnExitCommandListCreateImmediate(ze_command_list_create_immediate_params_t* params,
                                      ze_result_t result, void* global_data, void** instance_data) {
  std::cout << __PRETTY_FUNCTION__ << '\n';
  if (result == ze_result_t::ZE_RESULT_SUCCESS) {
    auto* collector = static_cast<ZeKernelViewCollector*>(global_data);
    if (!collector) {
      return;
    }
    collector->StoreCommandListMetadata(**(params->pphCommandList), *(params->phContext),
                                        *(params->phDevice), true);
  }
}

//void OnEnterEventCreate(ze_event_create_params_t* params, ze_result_t result, void* global_data,
//                         void** instance_data) {
//  std::cout << __PRETTY_FUNCTION__ << '\n';
//  std::cout << "Event Created with signal: " << (*(params->pdesc))->signal << '\n';
//  std::cout << "Event Created with wait: " << (*(params->pdesc))->wait << '\n';
//  ze_event_desc_t* edesc = new ze_event_desc_t{**(params->pdesc)};
//  edesc->wait |= ZE_EVENT_SCOPE_FLAG_HOST;
//  edesc->signal |= ZE_EVENT_SCOPE_FLAG_HOST;
//  *(params->pdesc) = edesc;
//}
// zeCommandListDestroy
void OnExitCommandListDestroy(ze_command_list_destroy_params_t* params, ze_result_t result,
                              void* global_data, void** instance_data) {
  std::cout << __PRETTY_FUNCTION__ << '\n';
}

// zeCommandListReset
void OnExitCommandListReset(ze_command_list_reset_params_t* params, ze_result_t result,
                            void* global_data, void** instance_data) {
  std::cout << __PRETTY_FUNCTION__ << '\n';
  //static_cast<ZeKernelViewCollector*>(global_data)->StoreCommandListMetadata(*(params->phCommandList), nullptr, nullptr, false);
}

// zeCommandQueueExecuteCommandLists
void OnExitCommandQueueExecuteCommandLists(ze_command_queue_execute_command_lists_params_t* params,
                                           ze_result_t result, void* global_data,
                                           void** instance_data) {
  std::cout << __PRETTY_FUNCTION__ << '\n';
}

// zeCommandQueueSychronize
void OnExitCommandQueueSynchronize(ze_command_queue_synchronize_params_t* params,
                                   ze_result_t result, void* global_data, void** instance_data) {
  std::cout << __PRETTY_FUNCTION__ << '\n';
  if(result == ze_result_t::ZE_RESULT_SUCCESS) {
    auto& ev_list = static_cast<ZeKernelViewCollector*>(global_data)->GetEventList();
    auto ev_list_it = std::begin(ev_list);
    while(ev_list_it != std::end(ev_list)) {
      result = zeEventQueryStatus(ev_list_it->event);
      if (result == ZE_RESULT_NOT_READY) {
        ++ev_list_it;
      } else if (result == ZE_RESULT_SUCCESS) {
        auto ts_result = std::move(ev_list_it->mem);
        auto ts_result_t = *static_cast<ze_kernel_timestamp_result_t*>(ts_result.Get());
        std::cout << "Timestamps found" << '\n';
        std::cout << "\tBegin: " << ts_result_t.global.kernelStart << '\n';
        std::cout << "\tEnd: " << ts_result_t.global.kernelEnd << '\n';
        ev_list.erase(ev_list_it);
      } else {
        std::cerr << "Event failed for some other reason..." << '\n';
      }
    }
  }
    //for (const auto& kernel_ts : static_cast<ZeKernelViewCollector*>(global_data)->GetEventList()) {
    //  ze_result_t status = zeEventQueryStatus(kernel_ts.event);
    //  if (status == ZE_RESULT_SUCCESS) {
      //}
}

// zeCommandQueueDestroy
void OnExitCommandQueueDestroy(ze_command_queue_destroy_params_t* params, ze_result_t result,
                               void* global_data, void** instance_data) {
  std::cout << __PRETTY_FUNCTION__ << '\n';
}

}  // namespace pti

#endif  // ZE_KERNEL_VIEW_COLLECTOR_H_
