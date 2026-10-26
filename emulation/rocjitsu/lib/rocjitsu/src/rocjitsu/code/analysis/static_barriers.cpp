// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/analysis/static_barriers.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/isa/instruction.h"

#include <array>
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
StaticBarrierMember make_barrier_member(const Instruction &instruction) {
  StaticBarrierMember result{instruction.src_loc(), std::vector<uint8_t>(instruction.size())};
  std::memcpy(result.bytes.data(), instruction.raw_encoding(), result.bytes.size());
  return result;
}
} // namespace

std::optional<StaticBarrierPolicy> static_barrier_policy(rj_code_target_id_t target,
                                                         const IsaTargetRegistry &registry) {
  const IsaTargetDescriptor *family = registry.find(target);
  if (!family || !registry.find_gpu_target(target))
    return std::nullopt;
  static constexpr std::array<uint32_t, 2> rdna_waves{32, 64};
  static constexpr std::array<uint32_t, 1> wave32{32};
  static constexpr std::array<uint32_t, 1> wave64{64};
  switch (family->architecture_id) {
  case ROCJITSU_CODE_ARCH_RDNA4:
    return StaticBarrierPolicy{StaticBarrierForm::BuiltinSplitPair, rdna_waves, -2};
  case ROCJITSU_CODE_ARCH_CDNA5:
    return StaticBarrierPolicy{StaticBarrierForm::BuiltinSplitPair, wave32, -4};
  case ROCJITSU_CODE_ARCH_CDNA3:
  case ROCJITSU_CODE_ARCH_CDNA4:
    return StaticBarrierPolicy{StaticBarrierForm::FullSingleton, wave64, -1};
  default:
    return std::nullopt;
  }
}

FailureOr<std::vector<StaticBarrier>> qualify_static_barriers(std::span<BasicBlock *const> blocks,
                                                              rj_code_target_id_t target,
                                                              StaticBarrierError &error) {
  error = {};
  const std::optional<StaticBarrierPolicy> policy = static_barrier_policy(target);
  if (!policy)
    return reject(error, StaticBarrierErrorCode::UnsupportedForm, 0,
                  "target has no complete-barrier mutation policy");
  std::vector<StaticBarrier> barriers;
  uint32_t clause_remaining = 0;
  for (BasicBlock *block : blocks) {
    std::optional<StaticBarrier> pending;
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
      if (in_clause || instruction.size() != 4)
        return reject(error, StaticBarrierErrorCode::UnsupportedForm, offset,
                      "barrier must be a complete instruction outside clauses");
      if (policy->form == StaticBarrierForm::FullSingleton) {
        if (name != "s_barrier")
          return reject(error, StaticBarrierErrorCode::UnsupportedForm, offset,
                        "only full s_barrier is qualified for this family");
        barriers.push_back({StaticBarrierForm::FullSingleton,
                            block->start_offset(),
                            -1,
                            StaticBarrierScope::Workgroup,
                            {make_barrier_member(instruction)}});
        continue;
      }
      if (name != "s_barrier_signal" && name != "s_barrier_wait")
        return reject(error, StaticBarrierErrorCode::UnsupportedForm, offset,
                      "only builtin immediate signal/wait pairs are qualified");
      const uint32_t word = instruction.raw_encoding()[0];
      if (name == "s_barrier_signal") {
        const uint32_t selector = word & 0xffu;
        if (selector < 193 || selector > 192u + static_cast<uint32_t>(-policy->minimum_builtin_id))
          return reject(error, StaticBarrierErrorCode::UnsupportedForm, offset,
                        "named, dynamic and literal barrier operands are outside this profile");
        if (pending)
          return reject(error, StaticBarrierErrorCode::IncompleteBarrier, offset,
                        "interleaved or repeated signal before completing wait");
        const int32_t id = -static_cast<int32_t>(selector - 192);
        pending =
            StaticBarrier{StaticBarrierForm::BuiltinSplitPair,
                          block->start_offset(),
                          id,
                          id >= -2 ? StaticBarrierScope::Workgroup : StaticBarrierScope::Cluster,
                          {make_barrier_member(instruction)}};
      } else {
        const int32_t id = static_cast<int16_t>(word & 0xffffu);
        if (!pending || pending->barrier_id != id)
          return reject(error, StaticBarrierErrorCode::IncompleteBarrier, offset,
                        "wait has no matching same-block signal");
        pending->members.push_back(make_barrier_member(instruction));
        barriers.push_back(std::move(*pending));
        pending.reset();
      }
    }
    if (pending)
      return reject(error, StaticBarrierErrorCode::IncompleteBarrier,
                    pending->members.front().offset, "signal has no completing same-block wait");
  }
  return barriers;
}
} // namespace rocjitsu
