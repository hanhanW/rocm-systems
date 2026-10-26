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
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rocjitsu {
namespace {

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

FailureOr<FaultBarrierInventory> resolve_selection(const FaultMutationContext &context,
                                                   const DropBarrierSelection &selection,
                                                   FaultMutationError &error) {
  error = {};
  if (selection.kind != FaultMutationKind::DropBarrier)
    return reject(error, FaultMutationErrorCode::InvalidParameters, "unknown mutation kind");
  if (!selection.belongs_to(context))
    return reject(error, FaultMutationErrorCode::BadIdentity,
                  "selection belongs to another or expired source context");
  auto inventory = discover_drop_barriers(context, selection.owner.entry, error);
  if (inventory.failed())
    return Result::failure();
  if (inventory.value().owner != selection.owner)
    return reject(error, FaultMutationErrorCode::BadIdentity, "owner facts changed");
  // Exact qualified membership below proves both form and cardinality; a
  // caller cannot turn a partial split pair into a full singleton by relabeling.
  if (std::ranges::count(inventory.value().barriers, selection.barrier) != 1)
    return reject(error, FaultMutationErrorCode::IncompleteSelection,
                  "exact complete barrier is absent or its qualification/bytes changed");
  return inventory;
}

} // namespace

FaultMutationContext::FaultMutationContext(std::span<const uint8_t> image)
    : source_(std::make_shared<const AmdGpuCodeObject>(image.data(), image.size())) {}

std::span<const uint8_t> FaultMutationContext::image() const {
  return {reinterpret_cast<const uint8_t *>(source_->image_data()), source_->image_size()};
}

bool DropBarrierSelection::belongs_to(const FaultMutationContext &context) const {
  return source_.lock().get() == &context.source();
}

DropBarrierSelection FaultBarrierInventory::select(const StaticBarrier &barrier) const {
  DropBarrierSelection selection;
  selection.owner = owner;
  selection.barrier = barrier;
  selection.source_ = source_;
  return selection;
}

std::vector<FaultMutationCapability>
fault_mutation_capabilities(const IsaTargetRegistry &registry) {
  std::vector<FaultMutationCapability> capabilities;
  for (const IsaTargetDescriptor &family : registry.targets()) {
    for (const IsaGpuTargetDescription &binding : family.gpu_targets) {
      const std::optional<StaticBarrierPolicy> policy =
          static_barrier_policy(binding.public_id, registry);
      if (!policy || !Decoder::create(registry, binding.public_id))
        continue;
      for (const uint32_t wave : policy->wave_sizes)
        capabilities.push_back(
            {FaultMutationKind::DropBarrier, policy->form, family.architecture_id,
             binding.public_id, wave,
             "code-object-v5; file-backed text/descriptors; exact kernel; "
             "independent siblings; closed direct CFG; complete builtin barrier"});
    }
  }
  return capabilities;
}

FailureOr<FaultBarrierInventory> discover_drop_barriers(const FaultMutationContext &context,
                                                        std::string_view entry,
                                                        FaultMutationError &error) {
  error = {};
  const auto image = context.image();
  const AmdGpuCodeObject &object = context.source();
  const rj_code_target_id_t target = object.target_id();
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
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const IsaTargetDescriptor *target_facts = registry.find(target);
  const IsaGpuTargetDescription *binding = registry.find_gpu_target(target);
  const std::optional<StaticBarrierPolicy> policy = static_barrier_policy(target, registry);
  if (header.e_ident[EI_ABIVERSION] != ELFABIVERSION_AMDGPU_HSA_V5 || !target_facts || !binding ||
      !policy || (header.e_flags & EF_AMDGPU_MACH) != binding->elf_machine ||
      (header.e_flags & ~uint32_t{0xfff}) != 0)
    return reject(error, FaultMutationErrorCode::UnsupportedTarget,
                  "requires a registered complete-barrier target and matching code-object-v5");
  if (!object.is_valid())
    return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid code object");
  if (object.text_sections().size() != 1)
    return reject(error, FaultMutationErrorCode::UnsupportedForm,
                  "requires one executable .text; multi-text is unsupported");
  const Section &text = *object.text_sections().front();
  if ((text.flags() & (SHF_ALLOC | SHF_EXECINSTR)) != (SHF_ALLOC | SHF_EXECINSTR))
    return reject(error, FaultMutationErrorCode::UnsupportedForm,
                  "selected .text must be allocated and executable");
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  const auto match_count = std::ranges::count(kernels, entry, &KernelDescriptorInfo::kernel_name);
  if (match_count > 1)
    return reject(error, FaultMutationErrorCode::AmbiguousSelection,
                  "exact entry name identifies multiple descriptors");
  if (entry.empty() || match_count == 0)
    return reject(error, FaultMutationErrorCode::BadIdentity,
                  "entry must match one exact kernel name");
  const KernelDescriptorInfo &kernel =
      *std::ranges::find(kernels, entry, &KernelDescriptorInfo::kernel_name);
  const uint32_t wave_size =
      kernel_wavefront_size(target_facts->architecture_id, kernel.descriptor);
  if (std::ranges::find(policy->wave_sizes, wave_size) == policy->wave_sizes.end() ||
      (policy->form == StaticBarrierForm::FullSingleton &&
       (kernel.descriptor.kernel_code_properties &
        amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) != 0))
    return reject(error, FaultMutationErrorCode::UnsupportedForm,
                  "descriptor wave mode is not admitted by this barrier family");

  const auto segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  if (!segments)
    return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid program headers");
  const auto text_address = file_backed_address(*segments, text.sectionOffset(), text.size(), PF_X);
  if (!text_address || *text_address != text.vaddr())
    return reject(error, FaultMutationErrorCode::UnsupportedForm,
                  "requires unambiguous file-backed executable .text at its declared address");
  for (const KernelDescriptorInfo &owner : kernels) {
    const auto descriptor_address =
        file_backed_address(*segments, owner.descriptor_file_offset, sizeof(owner.descriptor), 0);
    const auto loaded_entry =
        descriptor_address
            ? add_signed_offset(*descriptor_address, owner.descriptor.kernel_code_entry_byte_offset)
            : std::nullopt;
    // The scanner has already checked each section-based entry extent.
    if (!loaded_entry || *loaded_entry != text.vaddr() + owner.entry_text_offset)
      return reject(error, FaultMutationErrorCode::UnsupportedForm,
                    "descriptor load mapping does not resolve to its analyzed entry");
    if (kernarg_preload_length(owner.descriptor) != 0)
      return reject(error, FaultMutationErrorCode::UnsupportedForm,
                    "firmware preload ownership requires additional entry analysis");
  }

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
      // The scanner omits malformed descriptors. Every declared owner must
      // resolve before all-entry boundaries can establish exclusive ownership.
      if (symbol_name.ends_with(".kd")) {
        if (symbol.st_shndx >= sections.size() || symbol.st_size != sizeof(kernel.descriptor) ||
            elf_symbol_type(symbol.st_info) != kElfSymbolTypeObject ||
            elf_symbol_bind(symbol.st_info) != kElfSymbolBindGlobal)
          return reject(error, FaultMutationErrorCode::UnsupportedForm,
                        "unresolved descriptor owner");
        const Elf64_Shdr &owner = sections[symbol.st_shndx];
        if (owner.sh_type == SHT_NOBITS || symbol.st_value < owner.sh_addr ||
            symbol.st_value - owner.sh_addr > owner.sh_size ||
            symbol.st_size > owner.sh_size - (symbol.st_value - owner.sh_addr) ||
            owner.sh_offset > image.size() || owner.sh_size > image.size() - owner.sh_offset)
          return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid descriptor owner");
        const uint64_t file_offset = owner.sh_offset + (symbol.st_value - owner.sh_addr);
        if (!std::ranges::any_of(kernels, [&](const KernelDescriptorInfo &candidate) {
              return candidate.kernel_name == symbol_name.substr(0, symbol_name.size() - 3) &&
                     candidate.descriptor_file_offset == file_offset;
            }))
          return reject(error, FaultMutationErrorCode::UnsupportedForm,
                        "descriptor symbol was omitted from the owner inventory");
      }
      const uint8_t type = symbol.st_info & 0xfu;
      if (symbol.st_shndx != text.sectionHeaderIndex()) {
        if ((type == kElfSymbolTypeFunc || type == kElfSymbolTypeAmdGpuHsaKernel) &&
            symbol.st_shndx < sections.size() &&
            (sections[symbol.st_shndx].sh_flags & (SHF_ALLOC | SHF_EXECINSTR)) ==
                (SHF_ALLOC | SHF_EXECINSTR))
          return reject(error, FaultMutationErrorCode::UnsupportedForm,
                        "executable owners outside .text require broader ownership analysis");
        continue;
      }
      const bool external = (symbol.st_info >> 4) != kElfSymbolBindLocal;
      if (type == kElfSymbolTypeObject || type == kElfSymbolTypeAmdGpuHsaKernel ||
          ((type == kElfSymbolTypeFunc || external) &&
           (symbol.st_value < text.vaddr() ||
            !std::ranges::any_of(kernels, [&](const KernelDescriptorInfo &candidate) {
              return symbol.st_value - text.vaddr() == candidate.entry_text_offset;
            }))))
        return reject(error, FaultMutationErrorCode::UnsupportedForm,
                      "separate function/data owners inside .text are outside this profile");
    }
  }
  auto decoder = Decoder::create(registry, target);
  if (!decoder)
    return reject(error, FaultMutationErrorCode::UnsupportedTarget,
                  "concrete target decoder unavailable");
  std::vector<uint64_t> entries;
  std::unordered_set<uint64_t> all_entries;
  for (const auto &owner : kernels) {
    if (!all_entries.insert(owner.entry_text_offset).second)
      return reject(error, FaultMutationErrorCode::AmbiguousSelection,
                    "kernel descriptors alias one hardware entry");
    entries.push_back(owner.entry_text_offset);
  }
  util::StringDiagnostic diagnostic;
  auto blocks = BasicBlock::build_reachable(object, *decoder, target_facts->architecture_id,
                                            entries, diagnostic.emitter(), {}, {}, {}, target);
  if (blocks.failed())
    return reject(error, FaultMutationErrorCode::UnsupportedForm, diagnostic.message());
  const BlockOffsetIndex index = build_block_offset_index(blocks.value());
  const BlockPositionIndex positions = build_block_position_index(blocks.value());
  std::unordered_map<const BasicBlock *, uint64_t> block_owners;
  std::vector<BasicBlock *> selected_scope;
  for (const auto &owner : kernels) {
    BasicBlock *root = block_for_offset(index, owner.entry_text_offset);
    if (!root || root->start_offset() != owner.entry_text_offset)
      return reject(error, FaultMutationErrorCode::UnsupportedForm, "entry is not a block start");
    const KernelScopeSpec spec{all_entries, {owner.entry_text_offset}, {}};
    auto scope = reachable_kernel_blocks(blocks.value(), index, positions, *root, spec);
    for (BasicBlock *block : scope) {
      const auto [position, inserted] = block_owners.emplace(block, owner.entry_text_offset);
      if (!inserted && position->second != owner.entry_text_offset)
        return reject(error, FaultMutationErrorCode::UnsupportedForm,
                      "shared reachable blocks require broader ownership analysis",
                      block->start_offset());
      if (block->successor_issue() != BasicBlock::SuccessorIssue::None ||
          block->has_implicit_terminator() || !block->call_edges().empty())
        return reject(error, FaultMutationErrorCode::UnsupportedForm,
                      "kernel ownership requires a closed direct CFG", block->start_offset());
      for (const BasicBlock *successor : block->successors()) {
        if (all_entries.contains(successor->start_offset()) &&
            successor->start_offset() != owner.entry_text_offset)
          return reject(error, FaultMutationErrorCode::UnsupportedForm,
                        "control flow crosses another kernel entry", block->start_offset());
      }
      for (const Instruction &instruction : block->instructions()) {
        if ((instruction.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0 ||
            (is_control_flow_transfer(instruction) && !instruction.branch_offset_bytes()))
          return reject(error, FaultMutationErrorCode::UnsupportedForm,
                        "indirect owner is unsupported", instruction.src_loc());
      }
    }
    if (owner.descriptor_file_offset == kernel.descriptor_file_offset)
      selected_scope = std::move(scope);
  }
  if (block_owners.size() != blocks.value().size())
    return reject(error, FaultMutationErrorCode::UnsupportedForm, "unresolved code ownership");
  StaticBarrierError barrier_error;
  auto barriers = qualify_static_barriers(selected_scope, target, barrier_error);
  if (barriers.failed())
    return reject(error,
                  barrier_error.code == StaticBarrierErrorCode::IncompleteBarrier
                      ? FaultMutationErrorCode::IncompleteSelection
                      : FaultMutationErrorCode::UnsupportedForm,
                  barrier_error.message, barrier_error.offset);
  FaultBarrierInventory inventory;
  inventory.section = {*text.sectionHeaderIndex(), text.sectionOffset(), text.vaddr(), text.size()};
  inventory.owner = {kernel.kernel_name, kernel.entry_text_offset, kernel.descriptor_file_offset};
  inventory.barriers = std::move(barriers).value();
  inventory.source_ = context.source_;
  return inventory;
}

Result validate_drop_barrier_candidate(const FaultMutationContext &context,
                                       const DropBarrierSelection &selection,
                                       std::span<const uint8_t> candidate,
                                       FaultMutationError &error) {
  auto inventory = resolve_selection(context, selection, error);
  if (inventory.failed())
    return Result::failure();
  if (candidate.size() != context.image().size())
    return reject(error, FaultMutationErrorCode::ValidationFailure, "candidate size changed");
  auto decoder = Decoder::create(default_isa_target_registry(), context.source().target_id());
  std::vector<CheckedTextEdit> edits;
  // Membership comes from a new pristine inventory. Candidate bytes must decode
  // as canonical zero-cycle NOPs; this validator never invokes the NOP builder.
  const StaticBarrier &barrier = *std::ranges::find(inventory.value().barriers, selection.barrier);
  for (const StaticBarrierMember &member : barrier.members) {
    const size_t begin = inventory.value().section.file_offset + member.offset;
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
  TextEditError edit_error;
  if (validate_checked_text_candidate(context.source(), edits, candidate, edit_error).failed())
    return translate_edit_error(edit_error, error);
  const FaultMutationContext mutant(candidate);
  auto fresh = discover_drop_barriers(mutant, selection.owner.entry, error);
  if (fresh.failed())
    return reject(error, FaultMutationErrorCode::ValidationFailure,
                  "candidate failed fresh inventory: " + error.message, error.offset);
  std::vector<StaticBarrier> remaining = inventory.value().barriers;
  std::erase(remaining, barrier);
  if (fresh.value().barriers != remaining)
    return reject(error, FaultMutationErrorCode::ValidationFailure,
                  "candidate must remove exactly one logical barrier");
  return Result::success();
}

FailureOr<DropBarrierResult> apply_drop_barrier(const FaultMutationContext &context,
                                                const DropBarrierSelection &selection,
                                                FaultMutationError &error) {
  auto inventory = resolve_selection(context, selection, error);
  if (inventory.failed())
    return Result::failure();
  std::vector<CheckedTextEdit> edits;
  const auto *target = default_isa_target_registry().find(context.source().target_id());
  const uint32_t nop = build_s_nop(0, target->architecture_id);
  for (const StaticBarrierMember &member : selection.barrier.members) {
    CheckedTextEdit edit{member.offset, member.bytes, std::vector<uint8_t>(member.bytes.size())};
    std::memcpy(edit.replacement.data(), &nop, sizeof(nop));
    edits.push_back(std::move(edit));
  }
  TextEditError edit_error;
  auto applied = apply_checked_text_edits(context.source(), edits, edit_error);
  if (applied.failed())
    return translate_edit_error(edit_error, error);
  if (validate_drop_barrier_candidate(context, selection, applied.value().image, error).failed())
    return Result::failure();
  DropBarrierResult result;
  result.selection = selection;
  result.edit_result = std::move(applied).value();
  result.assumptions = {
      "Static mutation affects every executed occurrence of the selected logical barrier.",
      "Same-block membership does not prove barrier convergence.",
      "Execution and detector qualification have not been performed."};
  return result;
}
} // namespace rocjitsu
