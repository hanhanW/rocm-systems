// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/result.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {
class BasicBlock;

enum class StaticBarrierScope { Workgroup, Cluster };
enum class StaticBarrierErrorCode { None, UnsupportedForm, IncompleteGroup };
struct StaticBarrierError {
  StaticBarrierErrorCode code = StaticBarrierErrorCode::None;
  uint64_t offset = 0;
  std::string message;
};
struct StaticBarrierMember {
  uint64_t offset = 0;
  std::vector<uint8_t> bytes;
  bool operator==(const StaticBarrierMember &) const = default;
};
struct StaticBarrierGroup {
  uint64_t block_offset = 0;
  int32_t barrier_id = 0;
  StaticBarrierScope scope = StaticBarrierScope::Workgroup;
  std::vector<StaticBarrierMember> members;
  bool operator==(const StaticBarrierGroup &) const = default;
};

/// @brief Recognize complete same-block gfx1250 builtin signal/wait groups.
/// @details Accepts immediate builtin IDs -1/-2 (workgroup) and -3/-4 (cluster).
/// Named lifecycle, dynamic operands, isfirst (SCC side effect), interleaved
/// groups and cross-block pairs fail closed. This establishes static membership,
/// not convergence, runtime epochs, or a memory-order/race verdict. Ownership
/// and closed CFG checks belong to the caller; cycles do not disqualify a group.
[[nodiscard]] FailureOr<std::vector<StaticBarrierGroup>>
qualify_static_barrier_groups(std::span<BasicBlock *const> blocks, rj_code_target_id_t target,
                              StaticBarrierError &error);

} // namespace rocjitsu
