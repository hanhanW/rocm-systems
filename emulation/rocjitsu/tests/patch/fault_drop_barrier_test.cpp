// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../dbi_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"

#include <gtest/gtest.h>

#include <cstring>
#include <fstream>
#include <iterator>

namespace rocjitsu {
namespace {
constexpr rj_code_target_id_t kTarget = ROCJITSU_CODE_TARGET_GFX1250;
constexpr rj_code_arch_t kArch = ROCJITSU_CODE_ARCH_CDNA5;
constexpr uint32_t kSignal = 0xbe804ec1u;
constexpr uint32_t kWait = 0xbf94ffffu;
constexpr uint32_t kGuestWait = 0xbfc60000u;
constexpr uint32_t kMov = 0xbe800081u;
constexpr uint32_t kEnd = 0xbfb00000u;

std::vector<uint8_t> fixture(std::vector<uint32_t> words = {kGuestWait, kSignal, kMov, kWait,
                                                            kGuestWait, kEnd}) {
  std::vector<uint8_t> image = test::make_amdgpu_kernel_elf(
      words, 0, 3, EF_AMDGPU_MACH_AMDGCN_GFX1250, 0, 0, false, false, false, false, true);
  image[EI_DATA] = ELFDATA2LSB;
  image[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  return image;
}
DropBarrierRequest request_from(const FaultBarrierInventory &inventory, size_t group = 0) {
  return {FaultMutationKind::DropBarrier, FaultMutationProfile::Gfx1250Wave32StaticCompleteBarrier,
          inventory.identity, inventory.groups.at(group)};
}

template <typename Edit> void edit_segment(std::vector<uint8_t> &image, size_t index, Edit edit) {
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  const size_t offset = header.e_phoff + index * sizeof(Elf64_Phdr);
  ASSERT_LE(offset + sizeof(Elf64_Phdr), image.size());
  Elf64_Phdr segment{};
  std::memcpy(&segment, image.data() + offset, sizeof(segment));
  edit(segment);
  std::memcpy(image.data() + offset, &segment, sizeof(segment));
}

void expect_load_mapping_rejection(const std::vector<uint8_t> &image) {
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(fixture(), kTarget, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  DropBarrierRequest request = request_from(inventory.value());
  // Bind the altered headers so the digest check cannot mask the admission bug.
  request.identity.image_sha256 = text_edit_image_digest(image);
  std::vector<uint8_t> candidate = image;
  const uint32_t nop = build_s_nop(0, kArch);
  for (const StaticBarrierMember &member : request.group.members)
    std::memcpy(candidate.data() + request.identity.section.file_offset + member.offset, &nop, 4);
  EXPECT_TRUE(discover_drop_barriers(image, kTarget, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
  EXPECT_TRUE(apply_drop_barrier(image, request, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
  EXPECT_TRUE(validate_drop_barrier_candidate(image, request, candidate, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
}

TEST(FaultDropBarrier, RejectsTextNotFileBacked) {
  for (const uint64_t file_size : {0u, 4u}) {
    SCOPED_TRACE(file_size);
    std::vector<uint8_t> image = fixture();
    edit_segment(image, 0, [&](Elf64_Phdr &segment) { segment.p_filesz = file_size; });
    expect_load_mapping_rejection(image);
  }
}

TEST(FaultDropBarrier, RejectsDescriptorNotFileBacked) {
  for (const uint64_t file_size : {0u, 63u}) {
    SCOPED_TRACE(file_size);
    std::vector<uint8_t> image = fixture();
    edit_segment(image, 1, [&](Elf64_Phdr &segment) { segment.p_filesz = file_size; });
    expect_load_mapping_rejection(image);
  }
}

TEST(FaultDropBarrier, RejectsTextLoadAddressMismatch) {
  std::vector<uint8_t> image = fixture();
  edit_segment(image, 0, [](Elf64_Phdr &segment) {
    segment.p_vaddr += 0x2000;
    segment.p_paddr += 0x2000;
  });
  expect_load_mapping_rejection(image);
}

TEST(FaultDropBarrier, RejectsDescriptorLoadAddressMismatch) {
  std::vector<uint8_t> image = fixture();
  edit_segment(image, 1, [](Elf64_Phdr &segment) {
    segment.p_vaddr += 0x2000;
    segment.p_paddr += 0x2000;
  });
  expect_load_mapping_rejection(image);
}

TEST(FaultDropBarrier, RejectsMissingOrNonExecutableTextLoad) {
  for (const bool missing : {false, true}) {
    SCOPED_TRACE(missing);
    std::vector<uint8_t> image = fixture();
    edit_segment(image, 0, [&](Elf64_Phdr &segment) {
      if (missing)
        segment.p_type = PT_NOTE;
      else
        segment.p_flags &= ~PF_X;
    });
    expect_load_mapping_rejection(image);
  }
}

TEST(FaultDropBarrier, RejectsOverlappingLoadMappings) {
  for (const size_t index : {0u, 1u}) {
    for (const bool zero_fill : {false, true}) {
      SCOPED_TRACE(index);
      SCOPED_TRACE(zero_fill);
      std::vector<uint8_t> image = fixture();
      Elf64_Ehdr header{};
      std::memcpy(&header, image.data(), sizeof(header));
      Elf64_Phdr overlap{};
      std::memcpy(&overlap, image.data() + header.e_phoff + index * sizeof(overlap),
                  sizeof(overlap));
      overlap.p_vaddr += 4;
      overlap.p_paddr += 4;
      overlap.p_filesz = zero_fill ? 0 : 4;
      overlap.p_memsz = 4;
      overlap.p_align = 1;
      // The fixture reserves enough header padding for this third segment.
      ASSERT_LE(header.e_phoff + 3 * sizeof(overlap), 0x100u);
      header.e_phnum = 3;
      std::memcpy(image.data(), &header, sizeof(header));
      std::memcpy(image.data() + header.e_phoff + 2 * sizeof(overlap), &overlap, sizeof(overlap));
      expect_load_mapping_rejection(image);
    }
  }
}

TEST(FaultDropBarrier, AcceptsFileBackedRangesInsideLargerLoadSegments) {
  std::vector<uint8_t> image = fixture();
  for (const size_t index : {0u, 1u}) {
    edit_segment(image, index, [](Elf64_Phdr &segment) {
      segment.p_offset -= 4;
      segment.p_vaddr -= 4;
      segment.p_paddr -= 4;
      segment.p_filesz += 4;
      segment.p_memsz += 4;
    });
  }
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(image, kTarget, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const auto request = request_from(inventory.value());
  const auto result = apply_drop_barrier(image, request, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(
      validate_drop_barrier_candidate(image, request, result.value().edit_result.image, error)
          .succeeded())
      << error.message;
}

TEST(FaultDropBarrier, CompleteSeparatedPairPreservesGuestWaitsAndMetadata) {
  const std::vector<uint8_t> image = fixture();
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(image, kTarget, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().groups.size(), 1u);
  EXPECT_EQ(inventory.value().groups.front().members[0].offset, 4u);
  EXPECT_EQ(inventory.value().groups.front().members[1].offset, 12u);
  const DropBarrierRequest request = request_from(inventory.value());
  FailureOr<DropBarrierResult> result = apply_drop_barrier(image, request, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(result.value().logical_mutation_count, 1u);
  EXPECT_EQ(result.value().edit_result.edits.size(), 2u);
  EXPECT_NE(result.value().mutant_sha256, request.identity.image_sha256);
  EXPECT_EQ(result.value().mutant_sha256, text_edit_image_digest(result.value().edit_result.image));
  EXPECT_EQ(result.value().execution_validation, FaultQualification::Unverified);
  EXPECT_EQ(result.value().detector_qualification, FaultQualification::Unverified);
  std::vector<uint8_t> expected = image;
  const uint32_t nop = build_s_nop(0, kArch);
  for (const uint64_t offset : {4u, 12u})
    std::memcpy(expected.data() + inventory.value().identity.section.file_offset + offset, &nop, 4);
  EXPECT_EQ(result.value().edit_result.image, expected);
  EXPECT_TRUE(validate_drop_barrier_candidate(image, request, expected, error).succeeded())
      << error.message;
  FailureOr<FaultBarrierInventory> fresh =
      discover_drop_barriers(expected, kTarget, "test_kernel", error);
  ASSERT_TRUE(fresh.succeeded()) << error.message;
  EXPECT_TRUE(fresh.value().groups.empty());
  EXPECT_TRUE(apply_drop_barrier(expected, request, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST(FaultDropBarrier, ExactSecondGroupSelectionLeavesRedundantFirstBarrier) {
  const std::vector<uint8_t> image = fixture({kSignal, kWait, kSignal, kMov, kWait, kEnd});
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(image, kTarget, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().groups.size(), 2u);
  FailureOr<DropBarrierResult> result =
      apply_drop_barrier(image, request_from(inventory.value(), 1), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  FailureOr<FaultBarrierInventory> fresh =
      discover_drop_barriers(result.value().edit_result.image, kTarget, "test_kernel", error);
  ASSERT_TRUE(fresh.succeeded()) << error.message;
  ASSERT_EQ(fresh.value().groups.size(), 1u);
  EXPECT_EQ(fresh.value().groups.front(), inventory.value().groups.front());
  EXPECT_EQ(result.value().detector_qualification, FaultQualification::Unverified);
}

TEST(FaultDropBarrier, RejectsMissingDuplicateStaleHalfWrongOwnerAndQualification) {
  const std::vector<uint8_t> image = fixture();
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(image, kTarget, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  for (int mode = 0; mode < 13; ++mode) {
    SCOPED_TRACE(mode);
    DropBarrierRequest request = request_from(inventory.value());
    switch (mode) {
    case 0:
      request.group.members.clear();
      break;
    case 1:
      request.group.members.pop_back();
      break;
    case 2:
      request.group.members.push_back(request.group.members.front());
      break;
    case 3:
      request.group.members.back() = request.group.members.front();
      break;
    case 4:
      request.group.members.back().bytes[0] ^= 1;
      break;
    case 5:
      request.identity.owners.front().entry = "kernel";
      break;
    case 6:
      ++request.identity.owners.front().entry_offset;
      break;
    case 7:
      ++request.identity.owners.front().descriptor_file_offset;
      break;
    case 8:
      request.identity.image_sha256[0] ^= 1;
      break;
    case 9:
      ++request.identity.section.index;
      break;
    case 10:
      request.identity.target = ROCJITSU_CODE_TARGET_GFX1251;
      break;
    case 11:
      request.group.scope = StaticBarrierScope::Cluster;
      break;
    case 12:
      request.kind = static_cast<FaultMutationKind>(99);
      break;
    }
    EXPECT_TRUE(apply_drop_barrier(image, request, error).failed());
    EXPECT_NE(error.code, FaultMutationErrorCode::None);
  }
}

TEST(FaultDropBarrier, RejectsPartialInterleavedDynamicAndNamedLifecycle) {
  const std::vector<std::vector<uint32_t>> cases{{kSignal, kEnd},
                                                 {kWait, kEnd},
                                                 {kSignal, kSignal, kWait, kEnd},
                                                 {kSignal, 0xbf94fffeu, kEnd},
                                                 {0xbe804e7du, kWait, kEnd},
                                                 {0xbe804e81u, 0xbf940001u, kEnd},
                                                 {0xbe804fc1u, kWait, kEnd},
                                                 {kSignal, 0xbe8050c1u, kWait, kEnd}};
  for (const std::vector<uint32_t> &words : cases) {
    const std::vector<uint8_t> image = fixture(words);
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(image, kTarget, "test_kernel", error).failed());
    EXPECT_NE(error.code, FaultMutationErrorCode::None);
  }
}

TEST(FaultDropBarrier, SameBlockStaticGroupInCycleAffectsAllOccurrences) {
  const std::vector<uint8_t> image = fixture({kSignal, kMov, kWait, 0xbfa1fffcu, kEnd});
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(image, kTarget, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().groups.size(), 1u);
  FailureOr<DropBarrierResult> result =
      apply_drop_barrier(image, request_from(inventory.value()), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  ASSERT_FALSE(result.value().assumptions.empty());
  EXPECT_NE(result.value().assumptions.front().find("every executed occurrence"),
            std::string::npos);
  EXPECT_EQ(std::memcmp(result.value().edit_result.image.data() + 0x10c, image.data() + 0x10c, 8),
            0);
}

TEST(FaultDropBarrier, RejectsCrossBlockPairAndUnresolvedControlFlow) {
  for (const std::vector<uint32_t> &words :
       std::vector<std::vector<uint32_t>>{{kSignal, build_s_branch(0, kArch), kWait, kEnd},
                                          {kSignal, kWait, build_s_branch(100, kArch)},
                                          {kSignal, kWait, 0xbe802000u}}) {
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(fixture(words), kTarget, "test_kernel", error).failed());
  }
}

TEST(FaultDropBarrier, RejectsForeignHeadersAndNonexactEntry) {
  const std::vector<uint8_t> original = fixture();
  for (const size_t index :
       {size_t{EI_CLASS}, size_t{EI_DATA}, size_t{EI_OSABI}, size_t{EI_ABIVERSION}}) {
    std::vector<uint8_t> image = original;
    image[index] = 0;
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(image, kTarget, "test_kernel", error).failed());
  }
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(original, kTarget, "kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(discover_drop_barriers(original, ROCJITSU_CODE_TARGET_GFX1201, "test_kernel", error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedTarget);
}

TEST(FaultDropBarrier, IndependentValidatorRejectsHalfDropAndUnrelatedCorruption) {
  const std::vector<uint8_t> image = fixture();
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(image, kTarget, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const DropBarrierRequest request = request_from(inventory.value());
  FailureOr<DropBarrierResult> result = apply_drop_barrier(image, request, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  for (const size_t offset :
       {size_t{0x104}, size_t{0x10c}, size_t{0x108}, size_t{0x110}, size_t{0x118}, size_t{48}}) {
    std::vector<uint8_t> candidate = result.value().edit_result.image;
    candidate[offset] ^= 1;
    EXPECT_TRUE(validate_drop_barrier_candidate(image, request, candidate, error).failed())
        << offset;
  }
  EXPECT_TRUE(validate_drop_barrier_candidate(image, request, image, error).failed());
}

TEST(FaultDropBarrier, CapabilityDescribesTransformationOnly) {
  const std::span<const FaultMutationCapability> capabilities = fault_mutation_capabilities();
  ASSERT_EQ(capabilities.size(), 1u);
  EXPECT_EQ(capabilities.front().target, kTarget);
  EXPECT_EQ(capabilities.front().wave_size, 32u);
  EXPECT_FALSE(capabilities.front().admission.empty());
}

TEST(FaultDropBarrier, BuiltinIdsKeepDistinctStaticScopes) {
  for (uint32_t selector = 193; selector <= 196; ++selector) {
    const int32_t id = -static_cast<int32_t>(selector - 192);
    const std::vector<uint8_t> image =
        fixture({0xbe804e00u | selector, 0xbf940000u | static_cast<uint16_t>(id), kEnd});
    FaultMutationError error;
    FailureOr<FaultBarrierInventory> inventory =
        discover_drop_barriers(image, kTarget, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().groups.size(), 1u);
    EXPECT_EQ(inventory.value().groups.front().barrier_id, id);
    EXPECT_EQ(inventory.value().groups.front().scope,
              id >= -2 ? StaticBarrierScope::Workgroup : StaticBarrierScope::Cluster);
    EXPECT_TRUE(apply_drop_barrier(image, request_from(inventory.value()), error).succeeded())
        << error.message;
  }
}

TEST(FaultDropBarrier, RejectsMultipleOwnersAndMalformedOmittedDescriptor) {
  for (const bool malformed : {false, true}) {
    std::vector<uint8_t> image = test::make_gfx950_two_kernel_elf({kSignal, kWait, kEnd}, 0);
    Elf64_Ehdr header{};
    std::memcpy(&header, image.data(), sizeof(header));
    header.e_ident[EI_DATA] = ELFDATA2LSB;
    header.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
    header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
    std::memcpy(image.data(), &header, sizeof(header));
    const AmdGpuCodeObject object(image.data(), image.size());
    const CodeObjectPatcher patcher(object);
    for (const Elf64_Shdr &section : patcher.section_headers()) {
      if (section.sh_type != SHT_SYMTAB)
        continue;
      for (size_t ordinal = 1; ordinal <= 2; ++ordinal) {
        Elf64_Sym symbol{};
        std::memcpy(&symbol, image.data() + section.sh_offset + ordinal * sizeof(symbol),
                    sizeof(symbol));
        // Both descriptors belong to the .rodata section.
        const Elf64_Shdr owner = patcher.section_headers()[symbol.st_shndx];
        amdhsa::kernel_descriptor_t descriptor{};
        const size_t offset = owner.sh_offset + symbol.st_value - owner.sh_addr;
        std::memcpy(&descriptor, image.data() + offset, sizeof(descriptor));
        descriptor.kernel_code_properties |= amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
        std::memcpy(image.data() + offset, &descriptor, sizeof(descriptor));
        if (malformed && ordinal == 2) {
          symbol.st_size = 0; // scanner skips this owner; frontend must reject it.
          std::memcpy(image.data() + section.sh_offset + ordinal * sizeof(symbol), &symbol,
                      sizeof(symbol));
        }
      }
    }
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(image, kTarget, "test_kernel0", error).failed());
    EXPECT_EQ(error.code, malformed ? FaultMutationErrorCode::UnsupportedForm
                                    : FaultMutationErrorCode::AmbiguousSelection)
        << error.message;
  }
}

TEST(FaultDropBarrier, RejectsWrongWaveModeAndBarrierInsideClause) {
  std::vector<uint8_t> image = fixture();
  const AmdGpuCodeObject object(image.data(), image.size());
  const Section &text = *object.text_sections().front();
  const std::vector<KernelDescriptorInfo> descriptors =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  ASSERT_EQ(descriptors.size(), 1u);
  amdhsa::kernel_descriptor_t descriptor = descriptors.front().descriptor;
  descriptor.kernel_code_properties &= ~amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
  std::memcpy(image.data() + descriptors.front().descriptor_file_offset, &descriptor,
              sizeof(descriptor));
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(image, kTarget, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
  EXPECT_TRUE(discover_drop_barriers(fixture({0xbf850001u, kSignal, kWait, kEnd}), kTarget,
                                     "test_kernel", error)
                  .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
}

TEST(FaultDropBarrier, RejectsTrapBetweenSignalAndWait) {
  FaultMutationError error;
  const std::vector<uint8_t> image = fixture({kSignal, 0xbf900001u, kWait, kEnd});
  EXPECT_TRUE(discover_drop_barriers(image, kTarget, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
}

#ifdef RJ_FAULT_CORPUS_OBJECT
TEST(FaultDropBarrier, PinnedCompiledCorpusCompleteDrop) {
  std::ifstream input(RJ_FAULT_CORPUS_OBJECT, std::ios::binary);
  ASSERT_TRUE(input.is_open());
  const std::vector<uint8_t> image((std::istreambuf_iterator<char>(input)), {});
  ASSERT_FALSE(input.bad());
  FaultMutationError error;
  constexpr std::string_view entry = "_Z16fa_barrier_epochPKtPjj";
  FailureOr<FaultBarrierInventory> inventory = discover_drop_barriers(image, kTarget, entry, error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().groups.size(), 1u);
  const StaticBarrierGroup &group = inventory.value().groups.front();
  EXPECT_EQ(group.barrier_id, -1);
  EXPECT_EQ(group.scope, StaticBarrierScope::Workgroup);
  ASSERT_EQ(group.members.size(), 2u);
  EXPECT_GT(group.members[1].offset, group.members[0].offset + group.members[0].bytes.size());
  FailureOr<DropBarrierResult> result =
      apply_drop_barrier(image, request_from(inventory.value()), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(result.value().edit_result.edits.size(), 2u);
  EXPECT_TRUE(validate_drop_barrier_candidate(image, result.value().request,
                                              result.value().edit_result.image, error)
                  .succeeded())
      << error.message;
  std::ofstream output(RJ_FAULT_CORPUS_MUTANT, std::ios::binary | std::ios::trunc);
  ASSERT_TRUE(output.is_open());
  const std::vector<uint8_t> &mutant = result.value().edit_result.image;
  output.write(reinterpret_cast<const char *>(mutant.data()), mutant.size());
  output.close();
  ASSERT_TRUE(output.good());
}
#endif

} // namespace
} // namespace rocjitsu
