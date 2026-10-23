// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/analysis/static_barrier_groups.h"
#include "rocjitsu/code/patch/checked_text_edits.h"

#include <array>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {

enum class FaultMutationKind { DropBarrier };
enum class FaultMutationProfile { Gfx1250Wave32StaticCompleteBarrier };
enum class FaultMutationErrorCode {
  None,
  BadIdentity,
  UnsupportedTarget,
  UnsupportedForm,
  AmbiguousSelection,
  IncompleteSelection,
  InvalidParameters,
  ValidationFailure,
  PublicationFailure,
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
struct FaultMutationIdentity {
  std::array<uint8_t, 32> image_sha256{};
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  TextSectionIdentity section;
  std::vector<FaultMutationOwner> owners;
  bool operator==(const FaultMutationIdentity &) const = default;
};
struct FaultBarrierInventory {
  FaultMutationIdentity identity;
  std::vector<StaticBarrierGroup> groups;
};
struct DropBarrierRequest {
  FaultMutationKind kind = FaultMutationKind::DropBarrier;
  FaultMutationProfile profile = FaultMutationProfile::Gfx1250Wave32StaticCompleteBarrier;
  FaultMutationIdentity identity;
  StaticBarrierGroup group;
};
enum class FaultQualification { Unverified, Validated };
struct FaultMutationCapability {
  FaultMutationKind kind;
  FaultMutationProfile profile;
  rj_code_target_id_t target;
  uint32_t wave_size;
  std::string_view admission;
};
struct DropBarrierResult {
  DropBarrierRequest request;
  CheckedTextEditResult edit_result;
  std::array<uint8_t, 32> mutant_sha256{};
  uint32_t logical_mutation_count = 1;
  FaultQualification transformation_validation = FaultQualification::Validated;
  FaultQualification execution_validation = FaultQualification::Unverified;
  FaultQualification detector_qualification = FaultQualification::Unverified;
  std::vector<std::string> assumptions;
};

/// @brief Enumerate transformation admission; no execution or detector claim.
[[nodiscard]] std::span<const FaultMutationCapability> fault_mutation_capabilities();

/// @brief Inventory a single exact kernel in a code-object-v5 gfx1250 wave32 image.
/// @details Requires complete .text in a file-backed executable PT_LOAD at its
/// declared address, and a file-backed descriptor whose relative entry resolves
/// to the analyzed entry. Each range must have one file mapping and no overlapping
/// PT_LOAD memory ranges. This bounded admission does not qualify a particular
/// runtime loader, GPU execution or detector outcome.
[[nodiscard]] FailureOr<FaultBarrierInventory>
discover_drop_barriers(std::span<const uint8_t> image, rj_code_target_id_t target,
                       std::string_view entry, FaultMutationError &error);

/// @brief Resolve the full pristine identity again and drop exactly one group.
[[nodiscard]] FailureOr<DropBarrierResult> apply_drop_barrier(std::span<const uint8_t> image,
                                                              const DropBarrierRequest &request,
                                                              FaultMutationError &error);

/// @brief Independently derive allowed members and decode NOPs in the candidate.
/// @details Does not call the mutation emitter or accept its proof as authority.
[[nodiscard]] Result validate_drop_barrier_candidate(std::span<const uint8_t> pristine,
                                                     const DropBarrierRequest &request,
                                                     std::span<const uint8_t> candidate,
                                                     FaultMutationError &error);

} // namespace rocjitsu
