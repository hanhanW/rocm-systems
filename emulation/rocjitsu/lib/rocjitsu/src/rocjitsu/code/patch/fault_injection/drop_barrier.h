// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/analysis/static_barriers.h"
#include "rocjitsu/code/patch/checked_text_edits.h"

#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

enum class FaultMutationKind { DropBarrier };
enum class FaultMutationErrorCode {
  None,
  BadIdentity,
  UnsupportedTarget,
  UnsupportedForm,
  AmbiguousSelection,
  IncompleteSelection,
  InvalidParameters,
  ValidationFailure,
};
struct FaultMutationError {
  FaultMutationErrorCode code = FaultMutationErrorCode::None;
  uint64_t offset = 0;
  std::string message;
};
struct FaultMutationOwner {
  std::string entry;
  uint64_t entry_offset = 0;
  uint64_t descriptor_file_offset = 0;
  bool operator==(const FaultMutationOwner &) const = default;
};

class FaultBarrierInventory;

/// @brief Own an immutable source snapshot and its selection lifetime.
/// @details Copies the input once. Independently constructed contexts have distinct
/// identities even for identical bytes. The owned code object supplies the target;
/// changing the caller's input buffer does not change this snapshot. This is the
/// bounded FI01 frontend context, not a shared helper/indirect-analysis cache.
class FaultMutationContext {
public:
  explicit FaultMutationContext(std::span<const uint8_t> image);
  FaultMutationContext(const FaultMutationContext &) = delete;
  FaultMutationContext &operator=(const FaultMutationContext &) = delete;

  const AmdGpuCodeObject &source() const { return *source_; }
  std::span<const uint8_t> image() const;

private:
  const std::shared_ptr<const AmdGpuCodeObject> source_;
  friend FailureOr<FaultBarrierInventory>
  discover_drop_barriers(const FaultMutationContext &, std::string_view, FaultMutationError &);
};

/// @brief Select one complete logical barrier in its original context.
/// @details Value facts are re-resolved on apply, not trusted as proof. The private
/// weak binding rejects cross-context and expired selections without a digest,
/// raw pointer identity, or durable replay contract.
class DropBarrierSelection {
public:
  FaultMutationKind kind = FaultMutationKind::DropBarrier;
  FaultMutationOwner owner;
  StaticBarrier barrier;

  bool belongs_to(const FaultMutationContext &context) const;

private:
  std::weak_ptr<const AmdGpuCodeObject> source_;
  friend class FaultBarrierInventory;
};

class FaultBarrierInventory {
public:
  TextSectionIdentity section;
  FaultMutationOwner owner;
  std::vector<StaticBarrier> barriers;

  /// @brief Bind an explicitly chosen complete barrier; apply checks its exact current facts.
  DropBarrierSelection select(const StaticBarrier &barrier) const;

private:
  std::weak_ptr<const AmdGpuCodeObject> source_;
  friend FailureOr<FaultBarrierInventory>
  discover_drop_barriers(const FaultMutationContext &, std::string_view, FaultMutationError &);
};

enum class FaultQualification { Unverified, Validated };
struct FaultMutationCapability {
  FaultMutationKind kind;
  StaticBarrierForm form;
  rj_code_arch_t architecture;
  rj_code_target_id_t target;
  uint32_t wave_size;
  std::string_view admission;
};
struct DropBarrierResult {
  DropBarrierSelection selection;
  CheckedTextEditResult edit_result;
  uint32_t logical_mutation_count = 1;
  FaultQualification transformation_validation = FaultQualification::Validated;
  FaultQualification execution_validation = FaultQualification::Unverified;
  FaultQualification detector_qualification = FaultQualification::Unverified;
  std::vector<std::string> assumptions;
};

/// @brief Enumerate transformation admission; no execution or detector claim.
[[nodiscard]] std::vector<FaultMutationCapability>
fault_mutation_capabilities(const IsaTargetRegistry &registry = default_isa_target_registry());

/// @brief Inventory one exact kernel, allowing independent siblings in one .text.
/// @details Requires an admitted family/target/wave and code-object-v5, complete file-backed
/// executable .text at its declared address, and loaded descriptors resolving to the analyzed
/// entries. Reject ambiguous entry names, malformed descriptors, overlapping load
/// mappings and unresolved/shared ownership. Multi-text, helpers and indirect flow
/// remain outside this slice. Admission is not runtime or detector qualification.
/// Non-aliasing unselected executable sections are preserved; declared executable
/// function/kernel owners outside .text require broader ownership analysis.
[[nodiscard]] FailureOr<FaultBarrierInventory>
discover_drop_barriers(const FaultMutationContext &context, std::string_view entry,
                       FaultMutationError &error);

/// @brief Re-resolve a same-context selection and drop exactly one logical barrier.
[[nodiscard]] FailureOr<DropBarrierResult> apply_drop_barrier(const FaultMutationContext &context,
                                                              const DropBarrierSelection &selection,
                                                              FaultMutationError &error);

/// @brief Re-derive pristine members, decode candidate NOPs and inventory it afresh.
/// @details Does not call the mutation emitter or accept its proof as authority.
[[nodiscard]] Result validate_drop_barrier_candidate(const FaultMutationContext &context,
                                                     const DropBarrierSelection &selection,
                                                     std::span<const uint8_t> candidate,
                                                     FaultMutationError &error);

} // namespace rocjitsu
