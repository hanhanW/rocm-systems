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

class AmdGpuCodeObject;

enum class TextEditErrorCode {
  None,
  InvalidImage,
  UnsupportedTarget,
  UnsupportedReferences,
  InvalidRange,
  Overlap,
  StaleBytes,
  InvalidInstruction,
  Clause,
  ValidationFailure,
};

struct TextEditError {
  TextEditErrorCode code = TextEditErrorCode::None;
  uint64_t offset = 0;
  std::string message;
};

/// @brief Source section facts; offsets are relative to the supplied code object.
struct TextSectionIdentity {
  uint64_t index = 0;
  uint64_t file_offset = 0;
  uint64_t vaddr = 0;
  uint64_t size = 0;
  bool operator==(const TextSectionIdentity &) const = default;
};

/// @brief Replace one complete original instruction with equally many bytes.
struct CheckedTextEdit {
  uint64_t offset = 0;
  std::vector<uint8_t> expected;
  std::vector<uint8_t> replacement;
  bool operator==(const CheckedTextEdit &) const = default;
};

/// @brief An equal-length range map, including unchanged source bytes.
struct TextEditMapping {
  uint64_t source_offset = 0;
  uint64_t candidate_offset = 0;
  uint64_t size = 0;
};

struct CheckedTextEditResult {
  std::vector<uint8_t> image;
  TextSectionIdentity section;
  /// Concrete public binding when present; legacy ELF MACH identities remain
  /// in image unchanged even when no public target enum has been allocated.
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  std::vector<CheckedTextEdit> edits;
  std::vector<TextEditMapping> mapping;
};

/// @brief Preflight every edit, then apply to a private CodeObjectPatcher image.
/// @details Supports a single executable .text, complete 4/8/12-byte source
/// instructions and complete replacement instructions. No layout or ELF metadata
/// changes are allowed. Relocation writes into text and unsupported references,
/// clause members, and symbol endpoints inside an edit are rejected. This proves
/// section-level structural integrity only. PT_LOAD mapping, descriptor admission
/// and the semantic delta are consumer obligations; success is not a loader or
/// execution qualification. Section-only images may have no program headers.
/// Other non-aliasing sections, including executable sections, remain byte-for-byte
/// unchanged. This helper does not infer code ownership from section flags.
/// The source object is the sole authority for target and section facts. Concrete
/// bindings always select their concrete decoder. Legacy unbound ELF MACH values
/// may select an existing non-variant family decoder through arch_for_elf_mach;
/// unknown MACH values and unbound variant-aware families fail closed.
[[nodiscard]] FailureOr<CheckedTextEditResult>
apply_checked_text_edits(const AmdGpuCodeObject &source, std::span<const CheckedTextEdit> edits,
                         TextEditError &error);

/// @brief Independently reparse a candidate and verify exact edits and unchanged bytes.
/// @details Does not call the emitter. Also performs source preflight so an
/// untrusted list of edits cannot authorize partial instructions or stale bytes.
[[nodiscard]] Result validate_checked_text_candidate(const AmdGpuCodeObject &source,
                                                     std::span<const CheckedTextEdit> edits,
                                                     std::span<const uint8_t> candidate,
                                                     TextEditError &error);

} // namespace rocjitsu
