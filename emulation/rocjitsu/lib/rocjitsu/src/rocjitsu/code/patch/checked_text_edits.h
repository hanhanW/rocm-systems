// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/result.h"

#include <array>
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

/// @brief Durable section identity; offsets are never arena pointers or ordinals.
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
[[nodiscard]] FailureOr<CheckedTextEditResult>
apply_checked_text_edits(const AmdGpuCodeObject &source, rj_code_target_id_t target,
                         std::span<const CheckedTextEdit> edits, TextEditError &error);

/// @brief Independently reparse a candidate and verify exact edits and unchanged bytes.
/// @details Does not call the emitter. Also performs source preflight so an
/// untrusted list of edits cannot authorize partial instructions or stale bytes.
[[nodiscard]] Result validate_checked_text_candidate(const AmdGpuCodeObject &source,
                                                     rj_code_target_id_t target,
                                                     std::span<const CheckedTextEdit> edits,
                                                     std::span<const uint8_t> candidate,
                                                     TextEditError &error);

[[nodiscard]] std::array<uint8_t, 32> text_edit_image_digest(std::span<const uint8_t> image);

} // namespace rocjitsu
