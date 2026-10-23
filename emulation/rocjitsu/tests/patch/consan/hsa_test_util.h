// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include <array>
#include <string_view>

namespace rocjitsu::consan::test {

struct Agents {
  hsa_agent_t cpu{}, gpu{};
  std::string_view target;
};

inline Agents agents(std::string_view target) {
  Agents found{{}, {}, target};
  hsa_iterate_agents(
      [](hsa_agent_t agent, void *data) {
        auto &result = *static_cast<Agents *>(data);
        hsa_device_type_t type{};
        if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
          return HSA_STATUS_SUCCESS;
        if (type == HSA_DEVICE_TYPE_CPU)
          result.cpu = agent;
        if (type == HSA_DEVICE_TYPE_GPU) {
          hsa_isa_t isa{};
          uint32_t bytes = 0;
          if (hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa) == HSA_STATUS_SUCCESS &&
              hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &bytes) == HSA_STATUS_SUCCESS &&
              bytes > 0 && bytes <= 256) {
            std::array<char, 256> name{};
            if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, name.data()) == HSA_STATUS_SUCCESS) {
              std::string_view identity(name.data());
              constexpr std::string_view prefix = "amdgcn-amd-amdhsa--";
              if (identity.starts_with(prefix)) {
                identity.remove_prefix(prefix.size());
                identity = identity.substr(0, identity.find(':'));
                if (identity == result.target)
                  result.gpu = agent;
              }
            }
          }
        }
        return HSA_STATUS_SUCCESS;
      },
      &found);
  return found;
}

inline hsa_amd_memory_pool_t fine_pool(hsa_agent_t cpu) {
  hsa_amd_memory_pool_t found{};
  hsa_amd_agent_iterate_memory_pools(
      cpu,
      [](hsa_amd_memory_pool_t pool, void *data) {
        hsa_amd_segment_t segment{};
        uint32_t flags = 0;
        bool allowed = false;
        if (hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment) ==
                HSA_STATUS_SUCCESS &&
            segment == HSA_AMD_SEGMENT_GLOBAL &&
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags) ==
                HSA_STATUS_SUCCESS &&
            hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALLOWED,
                                         &allowed) == HSA_STATUS_SUCCESS &&
            allowed &&
            (flags & (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
                      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) ==
                (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
                 HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
          *static_cast<hsa_amd_memory_pool_t *>(data) = pool;
          return HSA_STATUS_INFO_BREAK;
        }
        return HSA_STATUS_SUCCESS;
      },
      &found);
  return found;
}

} // namespace rocjitsu::consan::test
