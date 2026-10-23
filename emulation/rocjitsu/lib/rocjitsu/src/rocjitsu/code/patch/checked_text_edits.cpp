// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/checked_text_edits.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/text_padding.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace rocjitsu {
namespace {

Result reject(TextEditError &error, TextEditErrorCode code, std::string message,
              uint64_t offset = 0) {
  error = {code, offset, std::move(message)};
  return Result::failure();
}

std::span<const uint8_t> image_bytes(const AmdGpuCodeObject &object) {
  return {reinterpret_cast<const uint8_t *>(object.image_data()), object.image_size()};
}

struct SourceDecoder {
  rj_code_arch_t architecture;
  std::unique_ptr<Decoder> decoder;
};

std::optional<SourceDecoder> source_decoder(const AmdGpuCodeObject &source) {
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const rj_code_target_id_t target = source.target_id();
  const IsaTargetDescriptor *descriptor = nullptr;
  std::unique_ptr<Decoder> decoder;
  if (target != ROCJITSU_CODE_TARGET_INVALID) {
    descriptor = registry.find(target);
    decoder = Decoder::create(registry, target);
  } else {
    if (source.image_size() < sizeof(Elf64_Ehdr))
      return std::nullopt;
    Elf64_Ehdr header{};
    std::memcpy(&header, source.image_data(), sizeof(header));
    descriptor = registry.find(arch_for_elf_mach(header.e_flags));
    // Legacy ELF identities can use a family decoder only when the provider
    // has no variant-specific decoder behavior. Never substitute a default
    // concrete target for an unavailable variant binding.
    if (!descriptor || descriptor->variant_decoder_factory)
      return std::nullopt;
    decoder = Decoder::create(registry, descriptor->architecture_id);
  }
  if (!descriptor || !decoder)
    return std::nullopt;
  return SourceDecoder{descriptor->architecture_id, std::move(decoder)};
}

bool reference_enters_edit(uint64_t offset, std::span<const CheckedTextEdit> edits) {
  const auto next = std::upper_bound(
      edits.begin(), edits.end(), offset,
      [](uint64_t value, const CheckedTextEdit &edit) { return value < edit.offset; });
  if (next == edits.begin())
    return false;
  const CheckedTextEdit &edit = *std::prev(next);
  return offset > edit.offset && offset - edit.offset < edit.expected.size();
}

Result decode_replacement(Decoder &decoder, std::span<const uint8_t> bytes, uint64_t offset,
                          TextEditError &error) {
  std::vector<uint32_t> words(bytes.size() / sizeof(uint32_t));
  std::memcpy(words.data(), bytes.data(), bytes.size());
  size_t cursor = 0;
  while (cursor < words.size()) {
    DecodeResult decoded = decoder.decode_window(std::span(words).subspan(cursor), offset);
    if (decoded.failed() || decoded.value()->size() == 0)
      return reject(error, TextEditErrorCode::InvalidInstruction,
                    "replacement contains an invalid or truncated instruction", offset);
    if (decoded.value()->mnemonic() == "s_clause")
      return reject(error, TextEditErrorCode::Clause, "replacement may not introduce a clause",
                    offset);
    cursor += decoded.value()->size() / sizeof(uint32_t);
  }
  return Result::success();
}

FailureOr<TextSectionIdentity> preflight(const AmdGpuCodeObject &source,
                                         std::span<const CheckedTextEdit> edits,
                                         TextEditError &error) {
  error = {};
  if (!source.is_valid() || source.text_sections().size() != 1)
    return reject(error, TextEditErrorCode::InvalidImage,
                  "expected one allocated executable .text section");
  const Section &text = *source.text_sections().front();
  if ((text.flags() & (SHF_ALLOC | SHF_EXECINSTR)) != (SHF_ALLOC | SHF_EXECINSTR) ||
      !text.sectionHeaderIndex() || text.size() == 0 || text.size() % 4 != 0 ||
      text.vaddr() > std::numeric_limits<uint64_t>::max() - text.size() ||
      text.sectionOffset() > source.image_size() ||
      text.size() > source.image_size() - text.sectionOffset())
    return reject(error, TextEditErrorCode::InvalidImage, "invalid .text extent");
  std::optional<SourceDecoder> isa = source_decoder(source);
  if (!isa)
    return reject(error, TextEditErrorCode::UnsupportedTarget,
                  "source requires a concrete decoder or a known non-variant ELF architecture");
  Decoder &decoder = *isa->decoder;
  CodeObjectPatcher patcher(source);
  const std::vector<Elf64_Shdr> sections = patcher.section_headers();
  const auto aliases_text = [&](uint64_t offset, uint64_t size) {
    return size != 0 && offset < text.sectionOffset() + text.size() &&
           (offset >= text.sectionOffset() || size > text.sectionOffset() - offset);
  };
  Elf64_Ehdr header{};
  std::memcpy(&header, source.image_data(), sizeof(header));
  if (aliases_text(0, sizeof(header)) ||
      aliases_text(header.e_shoff, uint64_t{header.e_shnum} * header.e_shentsize) ||
      aliases_text(header.e_phoff, uint64_t{header.e_phnum} * header.e_phentsize))
    return reject(error, TextEditErrorCode::InvalidImage, ".text aliases ELF headers");
  for (size_t index = 0; index < sections.size(); ++index) {
    const Elf64_Shdr &section = sections[index];
    if (index != *text.sectionHeaderIndex() && section.sh_type != SHT_NULL &&
        section.sh_type != SHT_NOBITS && aliases_text(section.sh_offset, section.sh_size))
      return reject(error, TextEditErrorCode::InvalidImage, ".text aliases another section");
    // Relocatable sections have section-relative coordinates, commonly all
    // zero; only linked images share the virtual address space checked here.
    if (header.e_type != ET_REL && index != *text.sectionHeaderIndex() &&
        (section.sh_flags & SHF_ALLOC) != 0 && section.sh_size != 0 &&
        section.sh_addr < text.vaddr() + text.size() &&
        (section.sh_addr >= text.vaddr() || section.sh_size > text.vaddr() - section.sh_addr))
      return reject(error, TextEditErrorCode::InvalidImage,
                    ".text virtual addresses alias another allocated section");
  }
  if (patcher.has_relocations_within_text() || patcher.has_malformed_rocr_relocation_section() ||
      patcher.has_rocr_rejected_none_relocation() || patcher.has_unsupported_relocation_to_text())
    return reject(error, TextEditErrorCode::UnsupportedReferences,
                  "unsupported relocation place or reference to .text");
  if (edits.empty())
    return reject(error, TextEditErrorCode::InvalidRange, "at least one edit is required");

  // Validate arithmetic before constructing subspans or adding offsets. All
  // consumers use the same ordering for duplicate/overlap detection and commit.
  uint64_t previous_end = 0;
  for (const CheckedTextEdit &edit : edits) {
    const size_t size = edit.expected.size();
    if (edit.offset % 4 != 0 || (size != 4 && size != 8 && size != 12) ||
        edit.replacement.size() != size || edit.offset > text.size() ||
        size > text.size() - edit.offset)
      return reject(error, TextEditErrorCode::InvalidRange,
                    "edit requires an in-bounds, aligned, equal-length 4/8/12-byte extent",
                    edit.offset);
    if (edit.offset < previous_end)
      return reject(error, TextEditErrorCode::Overlap, "duplicate or overlapping edits",
                    edit.offset);
    previous_end = edit.offset + size;
    if (std::memcmp(text.data() + edit.offset, edit.expected.data(), size) != 0)
      return reject(error, TextEditErrorCode::StaleBytes, "expected bytes do not match source",
                    edit.offset);
  }

  // A streaming decode proves starts/ends without constructing a second CFG or
  // retaining every instruction. The padding policy is shared with BasicBlock.
  // Ownership and complete control flow remain consumer obligations, including
  // Instrumentor's new branches into caves that are not appended yet.
  const std::span<const uint32_t> words(reinterpret_cast<const uint32_t *>(text.data()),
                                        text.size() / sizeof(uint32_t));
  size_t edit_index = 0;
  uint32_t clause_remaining = 0;
  size_t cursor = 0;
  while (cursor < words.size()) {
    if (is_text_padding_word(isa->architecture, words[cursor])) {
      ++cursor;
      continue;
    }
    util::StringDiagnostic diagnostic;
    DecodeResult decoded = decoder.decode_window(words.subspan(cursor), cursor * sizeof(uint32_t),
                                                 diagnostic.emitter());
    if (decoded.failed())
      return reject(error, TextEditErrorCode::InvalidInstruction, diagnostic.message(), cursor * 4);
    const Instruction &instruction = *decoded.value();
    cursor += static_cast<size_t>(instruction.size()) / sizeof(uint32_t);
    if (const std::optional<int64_t> displacement = instruction.branch_offset_bytes()) {
      const std::optional<uint64_t> destination =
          add_signed_offset(cursor * sizeof(uint32_t), *displacement);
      if (!destination || reference_enters_edit(*destination, edits))
        return reject(error, TextEditErrorCode::UnsupportedReferences,
                      "branch reference overflows or enters an edited instruction",
                      instruction.src_loc());
    }
    const bool in_clause = clause_remaining != 0;
    if (in_clause)
      --clause_remaining;
    const bool is_clause = instruction.mnemonic() == "s_clause";
    if (is_clause)
      clause_remaining = std::max(clause_remaining, (instruction.raw_encoding()[0] & 0x3fu) + 1u);
    if (edit_index == edits.size() || instruction.src_loc() < edits[edit_index].offset)
      continue;
    const CheckedTextEdit &edit = edits[edit_index];
    if (instruction.src_loc() != edit.offset ||
        static_cast<size_t>(instruction.size()) != edit.expected.size())
      return reject(error, TextEditErrorCode::InvalidInstruction,
                    "edit does not cover one complete decoded source instruction", edit.offset);
    if (in_clause || is_clause)
      return reject(error, TextEditErrorCode::Clause, "cannot edit a clause or its members",
                    edit.offset);
    if (decode_replacement(decoder, edit.replacement, edit.offset, error).failed())
      return Result::failure();
    ++edit_index;
  }
  if (edit_index != edits.size())
    return reject(error, TextEditErrorCode::InvalidInstruction,
                  "edit starts in padding or outside decoded instructions",
                  edits[edit_index].offset);

  // Fixed addresses preserve text references only at retained boundaries. A
  // symbol start/end in a replaced literal would become a new instruction entry.
  for (const Elf64_Shdr &section : sections) {
    if (section.sh_type != SHT_SYMTAB && section.sh_type != SHT_DYNSYM)
      continue;
    if (section.sh_entsize != sizeof(Elf64_Sym) || section.sh_size % sizeof(Elf64_Sym) != 0)
      return reject(error, TextEditErrorCode::UnsupportedReferences, "malformed symbol table");
    const std::optional<std::vector<Elf64_Sym>> symbols =
        read_elf_table<Elf64_Sym>(image_bytes(source), section.sh_offset,
                                  section.sh_size / sizeof(Elf64_Sym), section.sh_entsize);
    if (!symbols)
      return reject(error, TextEditErrorCode::UnsupportedReferences, "malformed symbol table");
    for (const Elf64_Sym &symbol : *symbols) {
      if (symbol.st_shndx != *text.sectionHeaderIndex())
        continue;
      if (symbol.st_value < text.vaddr() || symbol.st_value - text.vaddr() > text.size() ||
          symbol.st_size > text.size() - (symbol.st_value - text.vaddr()))
        return reject(error, TextEditErrorCode::UnsupportedReferences, "symbol leaves .text");
      const uint64_t start = symbol.st_value - text.vaddr();
      for (const uint64_t endpoint : {start, start + symbol.st_size}) {
        if (reference_enters_edit(endpoint, edits))
          return reject(error, TextEditErrorCode::UnsupportedReferences,
                        "symbol endpoint lies inside an edited instruction", endpoint);
      }
    }
  }
  return TextSectionIdentity{*text.sectionHeaderIndex(), text.sectionOffset(), text.vaddr(),
                             text.size()};
}

Result validate_bytes(const AmdGpuCodeObject &source, rj_code_target_id_t target,
                      const TextSectionIdentity &section, std::span<const CheckedTextEdit> edits,
                      std::span<const uint8_t> candidate, TextEditError &error) {
  if (candidate.size() != source.image_size())
    return reject(error, TextEditErrorCode::ValidationFailure, "candidate image size changed");
  const AmdGpuCodeObject reparsed(candidate.data(), candidate.size());
  if (!reparsed.is_valid() || reparsed.target_id() != target ||
      reparsed.text_sections().size() != 1)
    return reject(error, TextEditErrorCode::ValidationFailure, "candidate container is invalid");
  const std::span<const uint8_t> original = image_bytes(source);
  std::optional<SourceDecoder> isa = source_decoder(source);
  if (!isa)
    return reject(error, TextEditErrorCode::ValidationFailure, "source decoder unavailable");
  size_t cursor = 0;
  for (const CheckedTextEdit &edit : edits) {
    const size_t begin = section.file_offset + edit.offset;
    if (!std::equal(original.begin() + cursor, original.begin() + begin,
                    candidate.begin() + cursor) ||
        !std::equal(edit.replacement.begin(), edit.replacement.end(), candidate.begin() + begin))
      return reject(error, TextEditErrorCode::ValidationFailure,
                    "candidate differs from exact authorized edits", edit.offset);
    if (decode_replacement(*isa->decoder, candidate.subspan(begin, edit.replacement.size()),
                           edit.offset, error)
            .failed())
      return Result::failure();
    cursor = begin + edit.replacement.size();
  }
  if (!std::equal(original.begin() + cursor, original.end(), candidate.begin() + cursor))
    return reject(error, TextEditErrorCode::ValidationFailure,
                  "candidate changed unselected bytes");
  return Result::success();
}

std::vector<CheckedTextEdit> sorted_edits(std::span<const CheckedTextEdit> edits) {
  std::vector<CheckedTextEdit> sorted(edits.begin(), edits.end());
  std::ranges::sort(sorted, {}, &CheckedTextEdit::offset);
  return sorted;
}

} // namespace

FailureOr<CheckedTextEditResult> apply_checked_text_edits(const AmdGpuCodeObject &source,
                                                          std::span<const CheckedTextEdit> edits,
                                                          TextEditError &error) {
  const rj_code_target_id_t target = source.target_id();
  std::vector<CheckedTextEdit> sorted = sorted_edits(edits);
  FailureOr<TextSectionIdentity> section = preflight(source, sorted, error);
  if (section.failed())
    return Result::failure();
  CodeObjectPatcher patcher(source);
  for (const CheckedTextEdit &edit : sorted)
    std::memcpy(patcher.text_bytes().data() + edit.offset, edit.replacement.data(),
                edit.replacement.size());
  std::vector<uint8_t> candidate = std::move(patcher).emit();
  if (validate_bytes(source, target, section.value(), sorted, candidate, error).failed())
    return Result::failure();
  return CheckedTextEditResult{std::move(candidate),
                               section.value(),
                               target,
                               std::move(sorted),
                               {{0, 0, section.value().size}}};
}

Result validate_checked_text_candidate(const AmdGpuCodeObject &source,
                                       std::span<const CheckedTextEdit> edits,
                                       std::span<const uint8_t> candidate, TextEditError &error) {
  const rj_code_target_id_t target = source.target_id();
  const std::vector<CheckedTextEdit> sorted = sorted_edits(edits);
  FailureOr<TextSectionIdentity> section = preflight(source, sorted, error);
  if (section.failed())
    return Result::failure();
  return validate_bytes(source, target, section.value(), sorted, candidate, error);
}

} // namespace rocjitsu
