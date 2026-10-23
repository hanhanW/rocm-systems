// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/static_barrier_groups.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"

#include <cstring>
#include <optional>
#include <utility>

namespace rocjitsu {
namespace {
Result reject(StaticBarrierError &error, StaticBarrierErrorCode code, uint64_t offset,
              std::string message) {
  error = {code, offset, std::move(message)};
  return Result::failure();
}
StaticBarrierMember member(const Instruction &instruction) {
  StaticBarrierMember result{instruction.src_loc(), std::vector<uint8_t>(instruction.size())};
  std::memcpy(result.bytes.data(), instruction.raw_encoding(), result.bytes.size());
  return result;
}
} // namespace

FailureOr<std::vector<StaticBarrierGroup>>
qualify_static_barrier_groups(std::span<BasicBlock *const> blocks, rj_code_target_id_t target,
                              StaticBarrierError &error) {
  error = {};
  if (target != ROCJITSU_CODE_TARGET_GFX1250)
    return reject(error, StaticBarrierErrorCode::UnsupportedForm, 0, "requires gfx1250");
  std::vector<StaticBarrierGroup> groups;
  uint32_t clause_remaining = 0;
  for (BasicBlock *block : blocks) {
    std::optional<StaticBarrierGroup> pending;
    for (const Instruction &instruction : block->instructions()) {
      const std::string_view name = instruction.mnemonic();
      const bool in_clause = clause_remaining != 0;
      if (in_clause)
        --clause_remaining;
      if (name == "s_clause")
        clause_remaining = (instruction.raw_encoding()[0] & 0x3fu) + 1u;
      if (name.find("barrier") == std::string_view::npos && !instruction.is_barrier())
        continue;
      const uint64_t offset = instruction.src_loc();
      if (in_clause || instruction.size() != 4 ||
          (name != "s_barrier_signal" && name != "s_barrier_wait"))
        return reject(error, StaticBarrierErrorCode::UnsupportedForm, offset,
                      "only builtin immediate signal/wait groups are qualified");
      const uint32_t word = instruction.raw_encoding()[0];
      if (name == "s_barrier_signal") {
        const uint32_t selector = word & 0xffu;
        if (selector < 193 || selector > 196)
          return reject(error, StaticBarrierErrorCode::UnsupportedForm, offset,
                        "named, dynamic and literal barrier operands are outside this profile");
        if (pending)
          return reject(error, StaticBarrierErrorCode::IncompleteGroup, offset,
                        "interleaved or repeated signal before completing wait");
        const int32_t id = -static_cast<int32_t>(selector - 192);
        pending = StaticBarrierGroup{block->start_offset(),
                                     id,
                                     id >= -2 ? StaticBarrierScope::Workgroup
                                              : StaticBarrierScope::Cluster,
                                     {member(instruction)}};
      } else {
        const int32_t id = static_cast<int16_t>(word & 0xffffu);
        if (!pending || pending->barrier_id != id)
          return reject(error, StaticBarrierErrorCode::IncompleteGroup, offset,
                        "wait has no matching same-block signal");
        pending->members.push_back(member(instruction));
        groups.push_back(std::move(*pending));
        pending.reset();
      }
    }
    if (pending)
      return reject(error, StaticBarrierErrorCode::IncompleteGroup, pending->members.front().offset,
                    "signal has no completing same-block wait");
  }
  return groups;
}
} // namespace rocjitsu
