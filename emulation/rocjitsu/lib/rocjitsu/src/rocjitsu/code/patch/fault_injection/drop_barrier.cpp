// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/control_flow.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/kernel_scope.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <utility>

namespace rocjitsu {
namespace {

constexpr FaultMutationCapability kCapabilities[] = {
    {FaultMutationKind::DropBarrier, FaultMutationProfile::Gfx1250Wave32StaticCompleteBarrier,
     ROCJITSU_CODE_TARGET_GFX1250, 32,
     "code-object-v5; file-backed text/descriptor; one kernel; closed direct CFG; "
     "same-block builtin signal/wait; IDs -1..-4"}};

Result reject(FaultMutationError &error, FaultMutationErrorCode code, std::string message,
              uint64_t offset = 0) {
  error = {code, offset, std::move(message)};
  return Result::failure();
}

Result translate_edit_error(const TextEditError &edit, FaultMutationError &error) {
  const bool unsupported = edit.code == TextEditErrorCode::UnsupportedReferences ||
                           edit.code == TextEditErrorCode::Clause;
  return reject(error,
                unsupported ? FaultMutationErrorCode::UnsupportedForm
                            : FaultMutationErrorCode::ValidationFailure,
                edit.message, edit.offset);
}

// Called only after AmdGpuCodeObject validates segment extents and VA arithmetic.
// Section payloads alone do not establish what a loader places at an address.
std::optional<uint64_t> file_backed_address(std::span<const Elf64_Phdr> segments, uint64_t offset,
                                            uint64_t size, uint32_t required_flags) {
  if (size == 0)
    return std::nullopt;
  const Elf64_Phdr *mapping = nullptr;
  uint64_t address = 0;
  for (const Elf64_Phdr &segment : segments) {
    if (segment.p_type != PT_LOAD || offset < segment.p_offset)
      continue;
    const uint64_t delta = offset - segment.p_offset;
    if (delta > segment.p_filesz || size > segment.p_filesz - delta)
      continue;
    if (mapping || (segment.p_flags & required_flags) != required_flags)
      return std::nullopt;
    mapping = &segment;
    address = segment.p_vaddr + delta;
  }
  if (!mapping)
    return std::nullopt;
  // Reject overlapping memory ranges, including zero-fill tails. Admission must
  // not depend on segment processing order or on a particular loader's policy.
  for (const Elf64_Phdr &segment : segments) {
    if (&segment != mapping && segment.p_type == PT_LOAD && segment.p_memsz != 0 &&
        segment.p_vaddr < address + size &&
        (segment.p_vaddr >= address || segment.p_memsz > address - segment.p_vaddr))
      return std::nullopt;
  }
  return address;
}

FailureOr<FaultBarrierInventory> resolve_request(std::span<const uint8_t> image,
                                                 const DropBarrierRequest &request,
                                                 FaultMutationError &error) {
  error = {};
  if (request.kind != FaultMutationKind::DropBarrier ||
      request.profile != FaultMutationProfile::Gfx1250Wave32StaticCompleteBarrier)
    return reject(error, FaultMutationErrorCode::InvalidParameters, "unknown kind or profile");
  if (request.identity.image_sha256 != text_edit_image_digest(image) ||
      request.identity.owners.size() != 1)
    return reject(error, FaultMutationErrorCode::BadIdentity,
                  "stale image digest or owner identity");
  FailureOr<FaultBarrierInventory> inventory = discover_drop_barriers(
      image, request.identity.target, request.identity.owners.front().entry, error);
  if (inventory.failed())
    return Result::failure();
  if (inventory.value().identity != request.identity)
    return reject(error, FaultMutationErrorCode::BadIdentity, "section, target or owner mismatch");
  if (request.group.members.size() != 2 ||
      request.group.members[0].offset >= request.group.members[1].offset)
    return reject(error, FaultMutationErrorCode::IncompleteSelection,
                  "request must contain two distinct members in signal/wait order");
  if (std::ranges::count(inventory.value().groups, request.group) != 1)
    return reject(error, FaultMutationErrorCode::IncompleteSelection,
                  "exact complete group is absent or its qualification/bytes changed");
  return inventory;
}

} // namespace

std::span<const FaultMutationCapability> fault_mutation_capabilities() { return kCapabilities; }

FailureOr<FaultBarrierInventory> discover_drop_barriers(std::span<const uint8_t> image,
                                                        rj_code_target_id_t target,
                                                        std::string_view entry,
                                                        FaultMutationError &error) {
  error = {};
  if (target != ROCJITSU_CODE_TARGET_GFX1250)
    return reject(error, FaultMutationErrorCode::UnsupportedTarget, "profile requires gfx1250");
  Elf64_Ehdr header{};
  if (image.size() < sizeof(header))
    return reject(error, FaultMutationErrorCode::UnsupportedForm, "truncated ELF header");
  std::memcpy(&header, image.data(), sizeof(header));
  if (std::memcmp(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_ident[EI_DATA] != ELFDATA2LSB ||
      header.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA || header.e_type != ET_DYN ||
      header.e_machine != EM_AMDGPU)
    return reject(error, FaultMutationErrorCode::UnsupportedForm,
                  "expected little-endian AMDHSA ET_DYN ELF64");
  if (header.e_ident[EI_ABIVERSION] != ELFABIVERSION_AMDGPU_HSA_V5 ||
      (header.e_flags & EF_AMDGPU_MACH) != EF_AMDGPU_MACH_AMDGCN_GFX1250 ||
      (header.e_flags & ~uint32_t{0xfff}) != 0)
    return reject(error, FaultMutationErrorCode::UnsupportedTarget,
                  "requires matching gfx1250 code-object-v5");
  const AmdGpuCodeObject object(image.data(), image.size());
  if (!object.is_valid() || object.target_id() != target || object.text_sections().size() != 1 ||
      object.allocated_executable_sections().size() != 1 ||
      object.allocated_executable_sections().front() != object.text_sections().front())
    return reject(error, FaultMutationErrorCode::UnsupportedForm,
                  "requires a valid image with one executable .text");
  const Section &text = *object.text_sections().front();
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  if (kernels.size() != 1)
    return reject(error, FaultMutationErrorCode::AmbiguousSelection,
                  "profile requires exactly one kernel descriptor");
  const KernelDescriptorInfo &kernel = kernels.front();
  if (entry.empty() || kernel.kernel_name != entry)
    return reject(error, FaultMutationErrorCode::BadIdentity,
                  "entry must match the exact kernel name");
  if ((kernel.descriptor.kernel_code_properties &
       amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) == 0 ||
      kernarg_preload_length(kernel.descriptor) != 0)
    return reject(error, FaultMutationErrorCode::UnsupportedForm,
                  "requires wave32 and one entry without firmware preload");

  const auto segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  if (!segments)
    return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid program headers");
  const auto text_address = file_backed_address(*segments, text.sectionOffset(), text.size(), PF_X);
  if (!text_address || *text_address != text.vaddr())
    return reject(error, FaultMutationErrorCode::UnsupportedForm,
                  "requires unambiguous file-backed executable .text at its declared address");
  const auto descriptor_address =
      file_backed_address(*segments, kernel.descriptor_file_offset, sizeof(kernel.descriptor), 0);
  const auto loaded_entry =
      descriptor_address
          ? add_signed_offset(*descriptor_address, kernel.descriptor.kernel_code_entry_byte_offset)
          : std::nullopt;
  // The descriptor scanner has already checked the section-based entry extent.
  if (!loaded_entry || *loaded_entry != text.vaddr() + kernel.entry_text_offset)
    return reject(error, FaultMutationErrorCode::UnsupportedForm,
                  "descriptor load mapping does not resolve to the selected entry");

  const CodeObjectPatcher patcher(object);
  if (patcher.has_relocations_within_text() || patcher.has_malformed_rocr_relocation_section() ||
      patcher.has_rocr_rejected_none_relocation() || patcher.has_unsupported_relocation_to_text())
    return reject(error, FaultMutationErrorCode::UnsupportedForm, "unsupported ELF references");
  const std::vector<Elf64_Shdr> sections = patcher.section_headers();
  for (const Elf64_Shdr &section : sections) {
    if (section.sh_type != SHT_SYMTAB && section.sh_type != SHT_DYNSYM)
      continue;
    if (section.sh_entsize != sizeof(Elf64_Sym) || section.sh_size % sizeof(Elf64_Sym) != 0)
      return reject(error, FaultMutationErrorCode::UnsupportedForm, "malformed symbol table");
    const std::optional<std::vector<Elf64_Sym>> symbols = read_elf_table<Elf64_Sym>(
        image, section.sh_offset, section.sh_size / sizeof(Elf64_Sym), section.sh_entsize);
    if (!symbols || section.sh_link >= sections.size())
      return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid symbol table range");
    const Elf64_Shdr &strings = sections[section.sh_link];
    if (strings.sh_type != SHT_STRTAB || strings.sh_offset > image.size() ||
        strings.sh_size > image.size() - strings.sh_offset)
      return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid symbol string table");
    for (const Elf64_Sym &symbol : *symbols) {
      if (symbol.st_name >= strings.sh_size)
        return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid symbol name");
      const char *name =
          reinterpret_cast<const char *>(image.data() + strings.sh_offset + symbol.st_name);
      const void *end = std::memchr(name, 0, strings.sh_size - symbol.st_name);
      if (!end)
        return reject(error, FaultMutationErrorCode::UnsupportedForm, "unterminated symbol name");
      const std::string_view symbol_name(name, static_cast<const char *>(end) - name);
      // The shared scanner intentionally omits malformed/out-of-text descriptors.
      // A mutation frontend must not turn that omission into a single-owner proof.
      if (symbol_name.ends_with(".kd")) {
        if (symbol_name != kernel.kernel_name + ".kd" || symbol.st_shndx >= sections.size() ||
            symbol.st_size != sizeof(kernel.descriptor))
          return reject(error, FaultMutationErrorCode::UnsupportedForm,
                        "unresolved descriptor owner");
        const Elf64_Shdr &owner = sections[symbol.st_shndx];
        if (symbol.st_value < owner.sh_addr || symbol.st_value - owner.sh_addr > owner.sh_size ||
            symbol.st_size > owner.sh_size - (symbol.st_value - owner.sh_addr) ||
            owner.sh_offset > image.size() || owner.sh_size > image.size() - owner.sh_offset ||
            owner.sh_offset + (symbol.st_value - owner.sh_addr) != kernel.descriptor_file_offset)
          return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid descriptor owner");
      }
      if (symbol.st_shndx != text.sectionHeaderIndex())
        continue;
      const uint8_t type = symbol.st_info & 0xfu;
      const bool external = (symbol.st_info >> 4) != kElfSymbolBindLocal;
      if (type == kElfSymbolTypeObject || type == kElfSymbolTypeAmdGpuHsaKernel ||
          ((type == kElfSymbolTypeFunc || external) &&
           (symbol.st_value < text.vaddr() ||
            symbol.st_value - text.vaddr() != kernel.entry_text_offset)))
        return reject(error, FaultMutationErrorCode::UnsupportedForm,
                      "separate function/data owners inside .text are outside this profile");
    }
  }
  std::unique_ptr<Decoder> decoder = Decoder::create(default_isa_target_registry(), target);
  if (!decoder)
    return reject(error, FaultMutationErrorCode::UnsupportedTarget, "gfx1250 decoder unavailable");
  const std::array<uint64_t, 1> entries{kernel.entry_text_offset};
  util::StringDiagnostic diagnostic;
  FailureOr<std::vector<std::unique_ptr<BasicBlock>>> blocks =
      BasicBlock::build_reachable(object, *decoder, ROCJITSU_CODE_ARCH_CDNA5, entries,
                                  diagnostic.emitter(), {}, {}, {}, target);
  if (blocks.failed())
    return reject(error, FaultMutationErrorCode::UnsupportedForm, diagnostic.message());
  const BlockOffsetIndex index = build_block_offset_index(blocks.value());
  BasicBlock *root = block_for_offset(index, kernel.entry_text_offset);
  if (!root || root->start_offset() != kernel.entry_text_offset)
    return reject(error, FaultMutationErrorCode::UnsupportedForm, "entry is not a block start");
  const KernelScopeSpec spec{{kernel.entry_text_offset}, {kernel.entry_text_offset}, {}};
  const std::vector<BasicBlock *> scope = reachable_kernel_blocks(
      blocks.value(), index, build_block_position_index(blocks.value()), *root, spec);
  if (scope.size() != blocks.value().size())
    return reject(error, FaultMutationErrorCode::UnsupportedForm, "unresolved code ownership");
  for (BasicBlock *block : scope) {
    if (block->successor_issue() != BasicBlock::SuccessorIssue::None ||
        block->has_implicit_terminator() || !block->call_edges().empty())
      return reject(error, FaultMutationErrorCode::UnsupportedForm, "CFG is not closed and direct",
                    block->start_offset());
    for (const Instruction &instruction : block->instructions()) {
      if ((instruction.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0 ||
          (is_control_flow_transfer(instruction) && !instruction.branch_offset_bytes()))
        return reject(error, FaultMutationErrorCode::UnsupportedForm,
                      "indirect owner is unsupported", instruction.src_loc());
    }
  }
  StaticBarrierError barrier_error;
  FailureOr<std::vector<StaticBarrierGroup>> groups =
      qualify_static_barrier_groups(scope, target, barrier_error);
  if (groups.failed())
    return reject(error,
                  barrier_error.code == StaticBarrierErrorCode::IncompleteGroup
                      ? FaultMutationErrorCode::IncompleteSelection
                      : FaultMutationErrorCode::UnsupportedForm,
                  barrier_error.message, barrier_error.offset);
  FaultMutationIdentity identity{
      text_edit_image_digest(image),
      target,
      {*text.sectionHeaderIndex(), text.sectionOffset(), text.vaddr(), text.size()},
      {{kernel.kernel_name, kernel.entry_text_offset, kernel.descriptor_file_offset}}};
  return FaultBarrierInventory{std::move(identity), std::move(groups).value()};
}

Result validate_drop_barrier_candidate(std::span<const uint8_t> pristine,
                                       const DropBarrierRequest &request,
                                       std::span<const uint8_t> candidate,
                                       FaultMutationError &error) {
  FailureOr<FaultBarrierInventory> inventory = resolve_request(pristine, request, error);
  if (inventory.failed())
    return Result::failure();
  if (candidate.size() != pristine.size())
    return reject(error, FaultMutationErrorCode::ValidationFailure, "candidate size changed");
  std::unique_ptr<Decoder> decoder =
      Decoder::create(default_isa_target_registry(), request.identity.target);
  std::vector<CheckedTextEdit> edits;
  // Membership comes from a new pristine inventory. Candidate bytes must decode
  // as canonical zero-cycle NOPs; this validator never invokes the NOP builder.
  const StaticBarrierGroup &group = *std::ranges::find(inventory.value().groups, request.group);
  for (const StaticBarrierMember &member : group.members) {
    const size_t begin = inventory.value().identity.section.file_offset + member.offset;
    uint32_t word = 0;
    std::memcpy(&word, candidate.data() + begin, sizeof(word));
    DecodeResult decoded = decoder->decode_window(std::span(&word, 1), member.offset);
    if (decoded.failed() || decoded.value()->size() != 4 ||
        decoded.value()->mnemonic() != "s_nop" || decoded.value()->num_src_operands() != 1 ||
        decoded.value()->src_operand(0)->encoding_value() != 0)
      return reject(error, FaultMutationErrorCode::ValidationFailure,
                    "each selected barrier member must become a complete zero-cycle NOP",
                    member.offset);
    edits.push_back({member.offset,
                     member.bytes,
                     {candidate.begin() + begin, candidate.begin() + begin + member.bytes.size()}});
  }
  const AmdGpuCodeObject source(pristine.data(), pristine.size());
  TextEditError edit_error;
  if (validate_checked_text_candidate(source, request.identity.target, edits, candidate, edit_error)
          .failed())
    return translate_edit_error(edit_error, error);
  FailureOr<FaultBarrierInventory> fresh = discover_drop_barriers(
      candidate, request.identity.target, request.identity.owners.front().entry, error);
  if (fresh.failed())
    return reject(error, FaultMutationErrorCode::ValidationFailure,
                  "candidate failed fresh inventory: " + error.message, error.offset);
  std::vector<StaticBarrierGroup> remaining = inventory.value().groups;
  std::erase(remaining, group);
  if (fresh.value().groups != remaining)
    return reject(error, FaultMutationErrorCode::ValidationFailure,
                  "candidate must remove exactly one logical group");
  return Result::success();
}

FailureOr<DropBarrierResult> apply_drop_barrier(std::span<const uint8_t> image,
                                                const DropBarrierRequest &request,
                                                FaultMutationError &error) {
  FailureOr<FaultBarrierInventory> inventory = resolve_request(image, request, error);
  if (inventory.failed())
    return Result::failure();
  std::vector<CheckedTextEdit> edits;
  const uint32_t nop = build_s_nop(0, ROCJITSU_CODE_ARCH_CDNA5);
  for (const StaticBarrierMember &member : request.group.members) {
    CheckedTextEdit edit{member.offset, member.bytes, std::vector<uint8_t>(member.bytes.size())};
    std::memcpy(edit.replacement.data(), &nop, sizeof(nop));
    edits.push_back(std::move(edit));
  }
  const AmdGpuCodeObject source(image.data(), image.size());
  TextEditError edit_error;
  FailureOr<CheckedTextEditResult> applied =
      apply_checked_text_edits(source, request.identity.target, edits, edit_error);
  if (applied.failed())
    return translate_edit_error(edit_error, error);
  if (validate_drop_barrier_candidate(image, request, applied.value().image, error).failed())
    return Result::failure();
  DropBarrierResult result;
  result.request = request;
  result.mutant_sha256 = text_edit_image_digest(applied.value().image);
  result.edit_result = std::move(applied).value();
  result.assumptions = {"Static mutation affects every executed occurrence of the selected group.",
                        "Same-block membership does not prove barrier convergence.",
                        "Execution and detector qualification have not been performed."};
  return result;
}
} // namespace rocjitsu
