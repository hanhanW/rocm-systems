// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "../dbi_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <optional>

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
DropBarrierSelection selection_from(const FaultBarrierInventory &inventory, size_t pair = 0) {
  return inventory.select(inventory.barriers.at(pair));
}

std::vector<uint8_t>
two_kernel_fixture(const std::vector<uint32_t> &first = {kSignal, kMov, kWait, kEnd},
                   const std::vector<uint32_t> &second = {kSignal, kMov, kWait, kEnd}) {
  std::vector<uint32_t> words(64, 0);
  std::copy(first.begin(), first.end(), words.begin());
  words.insert(words.end(), second.begin(), second.end());
  std::vector<uint8_t> image = test::make_gfx950_two_kernel_elf(words, 0);
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_ident[EI_DATA] = ELFDATA2LSB;
  header.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
  std::memcpy(image.data(), &header, sizeof(header));
  const AmdGpuCodeObject object(image.data(), image.size());
  const Section &text = *object.text_sections().front();
  const auto kernels =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  for (const auto &kernel : kernels) {
    auto descriptor = kernel.descriptor;
    descriptor.kernel_code_properties |= amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
    if (kernel.kernel_name == "test_kernel1")
      descriptor.kernel_code_entry_byte_offset += 256;
    std::memcpy(image.data() + kernel.descriptor_file_offset, &descriptor, sizeof(descriptor));
  }
  return image;
}

struct DropTargetRow {
  const char *name;
  rj_code_target_id_t target;
  rj_code_arch_t arch;
  uint32_t wave_size;
  bool split;
};
constexpr DropTargetRow kDropRows[] = {
    {"Gfx1200Wave32", ROCJITSU_CODE_TARGET_GFX1200, ROCJITSU_CODE_ARCH_RDNA4, 32, true},
    {"Gfx1200Wave64", ROCJITSU_CODE_TARGET_GFX1200, ROCJITSU_CODE_ARCH_RDNA4, 64, true},
    {"Gfx1201Wave32", ROCJITSU_CODE_TARGET_GFX1201, ROCJITSU_CODE_ARCH_RDNA4, 32, true},
    {"Gfx1201Wave64", ROCJITSU_CODE_TARGET_GFX1201, ROCJITSU_CODE_ARCH_RDNA4, 64, true},
    {"Gfx1250Wave32", ROCJITSU_CODE_TARGET_GFX1250, ROCJITSU_CODE_ARCH_CDNA5, 32, true},
    {"Gfx1251Wave32", ROCJITSU_CODE_TARGET_GFX1251, ROCJITSU_CODE_ARCH_CDNA5, 32, true},
    {"Gfx942Wave64", ROCJITSU_CODE_TARGET_GFX942, ROCJITSU_CODE_ARCH_CDNA3, 64, false},
    {"Gfx950Wave64", ROCJITSU_CODE_TARGET_GFX950, ROCJITSU_CODE_ARCH_CDNA4, 64, false},
};

std::vector<uint32_t> row_words(const DropTargetRow &row) {
  const uint32_t wait = row.split ? kGuestWait : 0xbf8c0000u;
  if (row.split)
    return {wait, kSignal, kMov, kWait, wait, kSignal, kMov, kWait, build_s_endpgm(row.arch)};
  return {wait, 0xbf8a0000u, kMov, wait, 0xbf8a0000u, build_s_endpgm(row.arch)};
}

std::vector<uint8_t> row_fixture(const DropTargetRow &row, std::vector<uint32_t> words,
                                 bool siblings = false) {
  const auto *binding = default_isa_target_registry().find_gpu_target(row.target);
  if (!binding)
    return {};
  if (siblings) {
    const auto body = words;
    words.resize(64, build_s_nop(0, row.arch));
    words.insert(words.end(), body.begin(), body.end());
  }
  auto image = siblings ? test::make_gfx950_two_kernel_elf(words, 0)
                        : test::make_amdgpu_kernel_elf(words, 0, 3, binding->elf_machine);
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_ident[EI_DATA] = ELFDATA2LSB;
  header.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  header.e_flags = binding->elf_machine;
  std::memcpy(image.data(), &header, sizeof(header));
  const AmdGpuCodeObject object(image.data(), image.size());
  const Section &text = *object.text_sections().front();
  const auto descriptors =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  for (const auto &kernel : descriptors) {
    auto descriptor = kernel.descriptor;
    if (row.wave_size == 32)
      descriptor.kernel_code_properties |= amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
    else
      descriptor.kernel_code_properties &= ~amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
    if (siblings && kernel.kernel_name == "test_kernel1")
      descriptor.kernel_code_entry_byte_offset += 256;
    std::memcpy(image.data() + kernel.descriptor_file_offset, &descriptor, sizeof(descriptor));
  }
  return image;
}

class FaultDropBarrierFamily : public ::testing::TestWithParam<DropTargetRow> {};

TEST_P(FaultDropBarrierFamily, CompleteLogicalBarrierPreservesTargetWaveAndOtherBarrier) {
  const DropTargetRow &row = GetParam();
  const auto image = row_fixture(row, row_words(row));
  const FaultMutationContext context(image);
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers.size(), 2u);
  const auto selection = inventory.value().select(inventory.value().barriers.back());
  const auto result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(result.value().logical_mutation_count, 1u);
  EXPECT_EQ(result.value().edit_result.edits.size(), row.split ? 2u : 1u);
  EXPECT_EQ(result.value().edit_result.target, row.target);
  auto expected = image;
  const uint32_t nop = build_s_nop(0, row.arch);
  for (const auto &member : selection.barrier.members)
    std::memcpy(expected.data() + inventory.value().section.file_offset + member.offset, &nop,
                sizeof(nop));
  EXPECT_EQ(result.value().edit_result.image, expected);
  EXPECT_TRUE(validate_drop_barrier_candidate(context, selection, expected, error).succeeded())
      << error.message;
  const FaultMutationContext mutant(expected);
  const auto remaining = discover_drop_barriers(mutant, "test_kernel", error);
  ASSERT_TRUE(remaining.succeeded()) << error.message;
  ASSERT_EQ(remaining.value().barriers.size(), 1u);
  EXPECT_EQ(remaining.value().barriers.front(), inventory.value().barriers.front());
  const Section &text = *mutant.source().text_sections().front();
  const auto kernels = scan_kernel_descriptors(mutant.image(), text.sectionOffset(), text.size(),
                                               text.sectionHeaderIndex());
  ASSERT_EQ(kernels.size(), 1u);
  EXPECT_EQ(kernel_wavefront_size(row.arch, kernels.front().descriptor), row.wave_size);
}

TEST_P(FaultDropBarrierFamily, ExactSiblingSelectionAndCandidateCorruption) {
  const DropTargetRow &row = GetParam();
  const auto image = row_fixture(row, row_words(row), true);
  const FaultMutationContext context(image);
  FaultMutationError error;
  for (const std::string entry : {"test_kernel0", "test_kernel1"}) {
    SCOPED_TRACE(entry);
    const auto inventory = discover_drop_barriers(context, entry, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    const auto sibling = discover_drop_barriers(
        context, entry == "test_kernel0" ? "test_kernel1" : "test_kernel0", error);
    ASSERT_TRUE(sibling.succeeded()) << error.message;
    const auto selection = inventory.value().select(inventory.value().barriers.back());
    const auto result = apply_drop_barrier(context, selection, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    auto expected = image;
    const uint32_t nop = build_s_nop(0, row.arch);
    for (const auto &member : selection.barrier.members)
      std::memcpy(expected.data() + inventory.value().section.file_offset + member.offset, &nop,
                  sizeof(nop));
    EXPECT_EQ(result.value().edit_result.image, expected);
    const FaultMutationContext mutant(expected);
    const auto fresh_sibling = discover_drop_barriers(mutant, sibling.value().owner.entry, error);
    ASSERT_TRUE(fresh_sibling.succeeded()) << error.message;
    EXPECT_EQ(fresh_sibling.value().barriers, sibling.value().barriers);
    const uint64_t text = inventory.value().section.file_offset;
    for (const uint64_t offset :
         {text + inventory.value().barriers.front().members.front().offset,
          text + inventory.value().owner.entry_offset,
          text + selection.barrier.members.front().offset,
          sibling.value().owner.descriptor_file_offset, uint64_t{offsetof(Elf64_Ehdr, e_flags)}}) {
      auto corrupted = expected;
      corrupted[offset] ^= 1;
      EXPECT_TRUE(validate_drop_barrier_candidate(context, selection, corrupted, error).failed())
          << offset;
      EXPECT_EQ(error.code, FaultMutationErrorCode::ValidationFailure);
    }
  }
}

TEST_P(FaultDropBarrierFamily, PreservesAdditionalExecutableSection) {
  const DropTargetRow &row = GetParam();
  auto image = row_fixture(row, row_words(row));
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  Elf64_Shdr section{};
  const size_t offset = header.e_shoff + 2 * sizeof(section);
  std::memcpy(&section, image.data() + offset, sizeof(section));
  section.sh_flags |= SHF_EXECINSTR;
  std::memcpy(image.data() + offset, &section, sizeof(section));
  const FaultMutationContext context(image);
  ASSERT_EQ(context.source().text_sections().size(), 1u);
  ASSERT_EQ(context.source().allocated_executable_sections().size(), 2u);
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const auto selection = selection_from(inventory.value());
  const auto result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  auto expected = image;
  const uint32_t nop = build_s_nop(0, row.arch);
  for (const auto &member : selection.barrier.members)
    std::memcpy(expected.data() + inventory.value().section.file_offset + member.offset, &nop, 4);
  EXPECT_EQ(result.value().edit_result.image, expected);
  EXPECT_TRUE(validate_drop_barrier_candidate(context, selection, expected, error).succeeded());
  expected[section.sh_offset] ^= 1;
  EXPECT_TRUE(validate_drop_barrier_candidate(context, selection, expected, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::ValidationFailure);
}

TEST_P(FaultDropBarrierFamily, RejectsForgedFormMembershipAndContext) {
  const DropTargetRow &row = GetParam();
  const auto image = row_fixture(row, row_words(row));
  const FaultMutationContext context(image);
  const FaultMutationContext other(image);
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const auto selected = selection_from(inventory.value());
  EXPECT_TRUE(apply_drop_barrier(other, selected, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  for (int mode = 0; mode < 5; ++mode) {
    auto forged = selected;
    switch (mode) {
    case 0:
      forged.barrier.members.pop_back();
      break;
    case 1:
      forged.barrier.members.push_back(forged.barrier.members.front());
      break;
    case 2:
      forged.barrier.members.front().bytes.front() ^= 1;
      break;
    case 3:
      forged.barrier.form =
          row.split ? StaticBarrierForm::FullSingleton : StaticBarrierForm::BuiltinSplitPair;
      forged.barrier.members.resize(1); // A half-pair still cannot become a singleton.
      break;
    case 4:
      ++forged.owner.descriptor_file_offset;
      break;
    }
    EXPECT_TRUE(apply_drop_barrier(context, forged, error).failed()) << mode;
    EXPECT_NE(error.code, FaultMutationErrorCode::None);
  }
}

TEST_P(FaultDropBarrierFamily, StaticBarrierInLoopAndFamilySpecificForms) {
  const DropTargetRow &row = GetParam();
  auto words = row_words(row);
  words.back() = build_s_branch(-static_cast<int32_t>(words.size()), row.arch);
  const FaultMutationContext context(row_fixture(row, words));
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  EXPECT_TRUE(apply_drop_barrier(context, selection_from(inventory.value()), error).succeeded())
      << error.message;
  if (row.split) {
    for (const uint32_t selector : {193u, 194u, 195u, 196u}) {
      const int32_t id = -static_cast<int32_t>(selector - 192);
      const FaultMutationContext builtin(
          row_fixture(row, {0xbe804e00u | selector, 0xbf940000u | static_cast<uint16_t>(id),
                            build_s_endpgm(row.arch)}));
      const auto discovered = discover_drop_barriers(builtin, "test_kernel", error);
      if (row.arch == ROCJITSU_CODE_ARCH_RDNA4 && selector > 194) {
        EXPECT_TRUE(discovered.failed());
        EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
      } else {
        ASSERT_TRUE(discovered.succeeded()) << error.message;
        ASSERT_EQ(discovered.value().barriers.size(), 1u);
        EXPECT_EQ(discovered.value().barriers.front().scope,
                  id >= -2 ? StaticBarrierScope::Workgroup : StaticBarrierScope::Cluster);
        EXPECT_TRUE(
            apply_drop_barrier(builtin, selection_from(discovered.value()), error).succeeded())
            << error.message;
      }
    }
    for (const std::vector<uint32_t> &incomplete : std::vector<std::vector<uint32_t>>{
             {kSignal, build_s_endpgm(row.arch)},
             {kWait, build_s_endpgm(row.arch)},
             {0xbe804fc1u, kWait, build_s_endpgm(row.arch)},
             {0xbe804e7du, kWait, build_s_endpgm(row.arch)},
             {0xbe804effu, 0xffffffffu, kWait, build_s_endpgm(row.arch)},
             {0xbe804e81u, 0xbf940001u, build_s_endpgm(row.arch)}}) {
      const FaultMutationContext unsupported(row_fixture(row, incomplete));
      EXPECT_TRUE(discover_drop_barriers(unsupported, "test_kernel", error).failed());
    }
  } else {
    DropTargetRow wrong_wave = row;
    wrong_wave.wave_size = 32;
    const FaultMutationContext unsupported(row_fixture(wrong_wave, row_words(row)));
    EXPECT_TRUE(discover_drop_barriers(unsupported, "test_kernel", error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
  }
}

TEST(FaultDropBarrier, RejectsUndecodableUnselectedRdnaPadding) {
  for (const DropTargetRow &row : kDropRows) {
    if (row.arch != ROCJITSU_CODE_ARCH_RDNA4)
      continue;
    auto image = row_fixture(row, row_words(row), true);
    const AmdGpuCodeObject source(image.data(), image.size());
    const uint32_t invalid = 0;
    std::memcpy(image.data() + source.text_sections().front()->sectionOffset() + 128, &invalid, 4);
    const FaultMutationContext context(image);
    FaultMutationError error;
    const auto inventory = discover_drop_barriers(context, "test_kernel0", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    EXPECT_TRUE(apply_drop_barrier(context, selection_from(inventory.value()), error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::ValidationFailure);
  }
}

TEST(FaultDropBarrier, ConcreteGfx1251DecoderSurvivesAllValidationStages) {
  const DropTargetRow &variant = kDropRows[5];
  auto words = row_words(variant);
  // Already used in InstrumentorProbePatch: gfx1251-only VOP3P, legal only with its features.
  words.insert(words.begin(), {0xcc4c4004u, 0x1a021908u});
  const auto image = row_fixture(variant, words);
  const FaultMutationContext context(image);
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const auto selection = selection_from(inventory.value());
  const auto result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(result.value().edit_result.target, ROCJITSU_CODE_TARGET_GFX1251);
  EXPECT_EQ(
      std::memcmp(result.value().edit_result.image.data() + inventory.value().section.file_offset,
                  image.data() + inventory.value().section.file_offset, 8),
      0);
  EXPECT_TRUE(
      validate_drop_barrier_candidate(context, selection, result.value().edit_result.image, error)
          .succeeded());
  const FaultMutationContext wrong_variant(row_fixture(kDropRows[4], words));
  EXPECT_TRUE(discover_drop_barriers(wrong_variant, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
}

TEST(FaultDropBarrier, RejectsFamiliesWithoutCompleteDropPolicyAndUnregisteredTargets) {
  for (const uint32_t machine :
       {EF_AMDGPU_MACH_AMDGCN_GFX908, EF_AMDGPU_MACH_AMDGCN_GFX90A, EF_AMDGPU_MACH_AMDGCN_GFX940,
        EF_AMDGPU_MACH_AMDGCN_GFX941, EF_AMDGPU_MACH_AMDGCN_GFX1010, EF_AMDGPU_MACH_AMDGCN_GFX1030,
        EF_AMDGPU_MACH_AMDGCN_GFX1100, EF_AMDGPU_MACH_AMDGCN_GFX1150, uint32_t{0xfe}}) {
    auto image = fixture();
    Elf64_Ehdr header{};
    std::memcpy(&header, image.data(), sizeof(header));
    header.e_flags = machine;
    std::memcpy(image.data(), &header, sizeof(header));
    const FaultMutationContext context(image);
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedTarget);
  }
}

INSTANTIATE_TEST_SUITE_P(TargetWaves, FaultDropBarrierFamily, ::testing::ValuesIn(kDropRows),
                         [](const ::testing::TestParamInfo<DropTargetRow> &info) {
                           return info.param.name;
                         });

TEST(FaultDropBarrier, ExactKernelSelectionPreservesIndependentSibling) {
  const auto image = two_kernel_fixture();
  for (const std::string entry : {"test_kernel0", "test_kernel1"}) {
    SCOPED_TRACE(entry);
    const FaultMutationContext context(image);
    FaultMutationError error;
    const auto inventory = discover_drop_barriers(context, entry, error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().barriers.size(), 1u);
    const std::string sibling_entry = entry == "test_kernel0" ? "test_kernel1" : "test_kernel0";
    const auto sibling = discover_drop_barriers(context, sibling_entry, error);
    ASSERT_TRUE(sibling.succeeded()) << error.message;
    const auto selection = selection_from(inventory.value());
    const auto result = apply_drop_barrier(context, selection, error);
    ASSERT_TRUE(result.succeeded()) << error.message;
    auto expected = image;
    const uint32_t nop = build_s_nop(0, kArch);
    for (const auto &member : selection.barrier.members)
      std::memcpy(expected.data() + inventory.value().section.file_offset + member.offset, &nop,
                  sizeof(nop));
    EXPECT_EQ(result.value().edit_result.image, expected);
    const FaultMutationContext mutant(expected);
    const auto fresh = discover_drop_barriers(mutant, entry, error);
    const auto fresh_sibling = discover_drop_barriers(mutant, sibling_entry, error);
    ASSERT_TRUE(fresh.succeeded()) << error.message;
    ASSERT_TRUE(fresh_sibling.succeeded()) << error.message;
    EXPECT_TRUE(fresh.value().barriers.empty());
    EXPECT_EQ(fresh_sibling.value().owner, sibling.value().owner);
    EXPECT_EQ(fresh_sibling.value().barriers, sibling.value().barriers);
    for (const uint64_t offset :
         {sibling.value().owner.descriptor_file_offset,
          sibling.value().section.file_offset + sibling.value().owner.entry_offset}) {
      auto corrupted = expected;
      corrupted[offset] ^= 1;
      EXPECT_TRUE(validate_drop_barrier_candidate(context, selection, corrupted, error).failed());
      EXPECT_EQ(error.code, FaultMutationErrorCode::ValidationFailure);
    }
  }
}

TEST(FaultDropBarrier, UnselectedKernelBarrierFormDoesNotChangeExactSelection) {
  const auto image =
      two_kernel_fixture({kSignal, kMov, kWait, kEnd}, {0xbe804e81u, 0xbf940001u, kEnd});
  const FaultMutationContext context(image);
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(context, "test_kernel0", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  EXPECT_TRUE(apply_drop_barrier(context, selection_from(inventory.value()), error).succeeded())
      << error.message;
}

TEST(FaultDropBarrier, SelectionRejectsAnotherContextEvenWithIdenticalBytes) {
  const auto image = fixture();
  const FaultMutationContext context(image);
  const FaultMutationContext other(image);
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const auto selection = selection_from(inventory.value());
  const auto result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(apply_drop_barrier(other, selection, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(
      validate_drop_barrier_candidate(other, selection, result.value().edit_result.image, error)
          .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST(FaultDropBarrier, ExpiredSelectionAndInventoryCannotBindANewContext) {
  const auto image = fixture();
  std::optional<FaultMutationContext> context;
  context.emplace(image);
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(*context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const auto selection = selection_from(inventory.value());
  const auto result = apply_drop_barrier(*context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  context.reset();
  context.emplace(image); // Reuse the context's address, not its source lifetime.
  EXPECT_TRUE(apply_drop_barrier(*context, selection, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(apply_drop_barrier(*context, selection_from(inventory.value()), error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(
      validate_drop_barrier_candidate(*context, selection, result.value().edit_result.image, error)
          .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST(FaultDropBarrier, SourceSnapshotSurvivesCallerEditsButSelectionCannotReplayThem) {
  auto image = fixture();
  const auto original = image;
  const FaultMutationContext context(image);
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const auto selection = selection_from(inventory.value());
  const uint32_t changed_mov = 0xbe800082u;
  std::memcpy(image.data() + inventory.value().section.file_offset + 8, &changed_mov, 4);
  EXPECT_TRUE(std::ranges::equal(context.image(), original));
  const FaultMutationContext changed(image);
  EXPECT_TRUE(apply_drop_barrier(changed, selection, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  const auto result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(
      std::memcmp(result.value().edit_result.image.data() + 0x108, original.data() + 0x108, 4), 0);
}

TEST(FaultDropBarrier, ExactSelectionRejectsDuplicateNamesAndSiblingOwnerSubstitution) {
  auto image = two_kernel_fixture();
  const FaultMutationContext context(image);
  FaultMutationError error;
  const auto first = discover_drop_barriers(context, "test_kernel0", error);
  const auto second = discover_drop_barriers(context, "test_kernel1", error);
  ASSERT_TRUE(first.succeeded());
  ASSERT_TRUE(second.succeeded());
  auto selection = selection_from(first.value());
  selection.owner = second.value().owner;
  EXPECT_TRUE(apply_drop_barrier(context, selection, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::IncompleteSelection);
  EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  const CodeObjectPatcher patcher(context.source());
  for (const auto &section : patcher.section_headers()) {
    if (section.sh_type != SHT_SYMTAB)
      continue;
    Elf64_Sym first_symbol{}, second_symbol{};
    std::memcpy(&first_symbol, image.data() + section.sh_offset + sizeof(Elf64_Sym),
                sizeof(first_symbol));
    std::memcpy(&second_symbol, image.data() + section.sh_offset + 2 * sizeof(Elf64_Sym),
                sizeof(second_symbol));
    second_symbol.st_name = first_symbol.st_name;
    std::memcpy(image.data() + section.sh_offset + 2 * sizeof(Elf64_Sym), &second_symbol,
                sizeof(second_symbol));
  }
  const FaultMutationContext ambiguous(image);
  EXPECT_TRUE(discover_drop_barriers(ambiguous, "test_kernel0", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::AmbiguousSelection);
}

TEST(FaultDropBarrier, RejectsCrossKernelEntryAndInteriorBranches) {
  for (const uint64_t destination : {256u, 260u}) {
    SCOPED_TRACE(destination);
    const auto image = two_kernel_fixture(
        {kSignal, kWait, build_s_branch(static_cast<int32_t>((destination - 12) / 4), kArch)});
    const FaultMutationContext context(image);
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(context, "test_kernel0", error).failed());
    EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
  }
}

TEST(FaultDropBarrier, RejectsSharedReachableBlocksAndUnresolvedSibling) {
  std::vector<uint32_t> first(35, 0);
  first[0] = build_s_branch(31, kArch); // Both entries reach the pair at .text+128.
  first[32] = kSignal;
  first[33] = kWait;
  first[34] = kEnd;
  const FaultMutationContext shared(two_kernel_fixture(first, {build_s_branch(-33, kArch)}));
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(shared, "test_kernel0", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
  const FaultMutationContext unresolved(two_kernel_fixture({kSignal, kWait, kEnd}, {0xbe802000u}));
  EXPECT_TRUE(discover_drop_barriers(unresolved, "test_kernel0", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
}

TEST(FaultDropBarrier, IndependentSiblingMalformedDescriptorCannotDisappearFromOwners) {
  auto image = two_kernel_fixture();
  const AmdGpuCodeObject object(image.data(), image.size());
  const CodeObjectPatcher patcher(object);
  for (const auto &section : patcher.section_headers()) {
    if (section.sh_type != SHT_SYMTAB)
      continue;
    Elf64_Sym symbol{};
    const auto offset = section.sh_offset + 2 * sizeof(symbol);
    std::memcpy(&symbol, image.data() + offset, sizeof(symbol));
    symbol.st_size = 0;
    std::memcpy(image.data() + offset, &symbol, sizeof(symbol));
  }
  const FaultMutationContext context(image);
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(context, "test_kernel0", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
}

TEST(FaultDropBarrier, PreservesUnselectedSectionButRejectsActualTextRelocations) {
  const auto original = two_kernel_fixture();
  const AmdGpuCodeObject source(original.data(), original.size());
  CodeObjectPatcher patcher(source);
  Elf64_Rela relocation{};
  relocation.r_offset = source.text_sections().front()->vaddr();
  relocation.r_info = R_AMDGPU_ABS64;
  const auto payload =
      std::span(reinterpret_cast<const uint8_t *>(&relocation), sizeof(relocation));
  ASSERT_TRUE(patcher.append_nonalloc_section(".fi01.extra", payload));
  auto image = patcher.emit();
  const FaultMutationContext context(image);
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(context, "test_kernel0", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const auto selection = selection_from(inventory.value());
  const auto result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  auto expected = image;
  const uint32_t nop = build_s_nop(0, kArch);
  for (const auto &member : selection.barrier.members)
    std::memcpy(expected.data() + inventory.value().section.file_offset + member.offset, &nop, 4);
  EXPECT_EQ(result.value().edit_result.image, expected);
  const CodeObjectPatcher updated(context.source());
  auto sections = updated.section_headers();
  auto &extra = sections.back();
  extra.sh_type = SHT_RELA;
  extra.sh_info = inventory.value().section.index;
  extra.sh_entsize = sizeof(Elf64_Rela);
  for (size_t index = 0; index < sections.size(); ++index)
    if (sections[index].sh_type == SHT_SYMTAB)
      extra.sh_link = index;
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  std::memcpy(image.data() + header.e_shoff + (sections.size() - 1) * sizeof(extra), &extra,
              sizeof(extra));
  const FaultMutationContext referenced(image);
  EXPECT_TRUE(discover_drop_barriers(referenced, "test_kernel0", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
}

TEST(FaultDropBarrier, RejectsDeclaredExecutableOwnerOutsideAnalyzedText) {
  const auto original = fixture();
  const AmdGpuCodeObject source(original.data(), original.size());
  CodeObjectPatcher patcher(source);
  const size_t code_index = patcher.section_headers().size();
  const std::array<uint32_t, 2> words{build_s_nop(0, kArch), build_s_endpgm(kArch)};
  const std::array<uint8_t, 7> names{0, 'e', 'x', 't', 'r', 'a', 0};
  std::array<Elf64_Sym, 2> symbols{};
  symbols[1].st_name = 1;
  symbols[1].st_info = (kElfSymbolBindGlobal << 4) | kElfSymbolTypeFunc;
  symbols[1].st_shndx = static_cast<uint16_t>(code_index);
  symbols[1].st_value = 0x6000;
  symbols[1].st_size = sizeof(words);
  ASSERT_TRUE(patcher.append_nonalloc_section(
      ".fi01.code", {reinterpret_cast<const uint8_t *>(words.data()), sizeof(words)}));
  ASSERT_TRUE(patcher.append_nonalloc_section(".fi01.names", names));
  ASSERT_TRUE(patcher.append_nonalloc_section(
      ".fi01.symbols", {reinterpret_cast<const uint8_t *>(symbols.data()), sizeof(symbols)}));
  auto sections = patcher.section_headers();
  ASSERT_EQ(sections.size(), code_index + 3);
  sections[code_index].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  sections[code_index].sh_addr = symbols[1].st_value;
  sections[code_index + 1].sh_type = SHT_STRTAB;
  auto &symtab = sections[code_index + 2];
  symtab.sh_type = SHT_SYMTAB;
  symtab.sh_entsize = sizeof(Elf64_Sym);
  symtab.sh_link = static_cast<uint32_t>(code_index + 1);
  symtab.sh_info = 1;
  auto image = patcher.emit();
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  std::memcpy(image.data() + header.e_shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
  const FaultMutationContext context(image);
  ASSERT_TRUE(context.source().is_valid());
  ASSERT_EQ(context.source().allocated_executable_sections().size(), 2u);
  // The section transaction can preserve this layout without proving its owners.
  const Section &text = *context.source().text_sections().front();
  CheckedTextEdit edit{0, std::vector<uint8_t>(4), std::vector<uint8_t>(4)};
  std::memcpy(edit.expected.data(), text.data(), 4);
  std::memcpy(edit.replacement.data(), words.data(), 4);
  TextEditError edit_error;
  EXPECT_TRUE(
      apply_checked_text_edits(context.source(), std::span(&edit, 1), edit_error).succeeded())
      << edit_error.message;
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
  EXPECT_NE(error.message.find("executable owners outside .text"), std::string::npos);
}

TEST(FaultDropBarrier, MultiTextHasAnExplicitDiagnostic) {
  const auto image = fixture();
  const AmdGpuCodeObject source(image.data(), image.size());
  CodeObjectPatcher patcher(source);
  const uint32_t nop = build_s_nop(0, kArch);
  ASSERT_TRUE(patcher.append_nonalloc_section(
      ".text", std::span(reinterpret_cast<const uint8_t *>(&nop), sizeof(nop))));
  auto multiple = patcher.emit();
  const AmdGpuCodeObject appended(multiple.data(), multiple.size());
  auto sections = CodeObjectPatcher(appended).section_headers();
  auto &extra = sections.back();
  extra.sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  extra.sh_addr = 0x4000;
  Elf64_Ehdr header{};
  std::memcpy(&header, multiple.data(), sizeof(header));
  std::memcpy(multiple.data() + header.e_shoff + (sections.size() - 1) * sizeof(extra), &extra,
              sizeof(extra));
  const FaultMutationContext context(multiple);
  ASSERT_EQ(context.source().text_sections().size(), 2u);
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
  EXPECT_NE(error.message.find("multi-text"), std::string::npos);
}

TEST(FaultDropBarrier, LongSameBlockPairUsesCompleteMembersWithoutDistanceCutoff) {
  std::vector<uint32_t> words(36, build_s_nop(0, kArch));
  words[0] = kSignal;
  words[34] = kWait;
  words[35] = kEnd;
  const FaultMutationContext context(fixture(words));
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers.size(), 1u);
  const auto result = apply_drop_barrier(context, selection_from(inventory.value()), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(result.value().logical_mutation_count, 1u);
  EXPECT_EQ(result.value().edit_result.edits.size(), 2u);
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
  const FaultMutationContext malformed(image);
  EXPECT_TRUE(discover_drop_barriers(malformed, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm) << error.message;
  const FaultMutationContext pristine(fixture());
  const auto inventory = discover_drop_barriers(pristine, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const auto selection = selection_from(inventory.value());
  std::vector<uint8_t> candidate = image;
  const uint32_t nop = build_s_nop(0, kArch);
  for (const StaticBarrierMember &member : selection.barrier.members)
    std::memcpy(candidate.data() + inventory.value().section.file_offset + member.offset, &nop, 4);
  // No valid same-context selection can be obtained from failed discovery.
  EXPECT_TRUE(apply_drop_barrier(malformed, selection, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(validate_drop_barrier_candidate(malformed, selection, candidate, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  EXPECT_TRUE(validate_drop_barrier_candidate(pristine, selection, candidate, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::ValidationFailure);
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
  const FaultMutationContext context(image);
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const auto selection = selection_from(inventory.value());
  const auto result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_TRUE(
      validate_drop_barrier_candidate(context, selection, result.value().edit_result.image, error)
          .succeeded())
      << error.message;
}

TEST(FaultDropBarrier, CompleteSeparatedPairPreservesGuestWaitsAndMetadata) {
  const std::vector<uint8_t> image = fixture();
  const FaultMutationContext context(image);
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers.size(), 1u);
  EXPECT_EQ(inventory.value().barriers.front().members[0].offset, 4u);
  EXPECT_EQ(inventory.value().barriers.front().members[1].offset, 12u);
  const DropBarrierSelection selection = selection_from(inventory.value());
  FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(result.value().logical_mutation_count, 1u);
  EXPECT_EQ(result.value().edit_result.edits.size(), 2u);
  EXPECT_EQ(result.value().edit_result.target, context.source().target_id());
  EXPECT_EQ(result.value().execution_validation, FaultQualification::Unverified);
  EXPECT_EQ(result.value().detector_qualification, FaultQualification::Unverified);
  std::vector<uint8_t> expected = image;
  const uint32_t nop = build_s_nop(0, kArch);
  for (const uint64_t offset : {4u, 12u})
    std::memcpy(expected.data() + inventory.value().section.file_offset + offset, &nop, 4);
  EXPECT_EQ(result.value().edit_result.image, expected);
  EXPECT_TRUE(validate_drop_barrier_candidate(context, selection, expected, error).succeeded())
      << error.message;
  const FaultMutationContext mutant(expected);
  FailureOr<FaultBarrierInventory> fresh = discover_drop_barriers(mutant, "test_kernel", error);
  ASSERT_TRUE(fresh.succeeded()) << error.message;
  EXPECT_TRUE(fresh.value().barriers.empty());
  EXPECT_TRUE(apply_drop_barrier(mutant, selection, error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
}

TEST(FaultDropBarrier, ExactSecondPairSelectionLeavesRedundantFirstBarrier) {
  const std::vector<uint8_t> image = fixture({kSignal, kWait, kSignal, kMov, kWait, kEnd});
  const FaultMutationContext context(image);
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers.size(), 2u);
  FailureOr<DropBarrierResult> result =
      apply_drop_barrier(context, selection_from(inventory.value(), 1), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  FailureOr<FaultBarrierInventory> fresh = discover_drop_barriers(
      FaultMutationContext(result.value().edit_result.image), "test_kernel", error);
  ASSERT_TRUE(fresh.succeeded()) << error.message;
  ASSERT_EQ(fresh.value().barriers.size(), 1u);
  EXPECT_EQ(fresh.value().barriers.front(), inventory.value().barriers.front());
  EXPECT_EQ(result.value().detector_qualification, FaultQualification::Unverified);
}

TEST(FaultDropBarrier, RejectsMissingDuplicateStaleHalfWrongOwnerAndQualification) {
  const std::vector<uint8_t> image = fixture();
  const FaultMutationContext context(image);
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  for (int mode = 0; mode < 13; ++mode) {
    SCOPED_TRACE(mode);
    DropBarrierSelection selection = selection_from(inventory.value());
    switch (mode) {
    case 0:
      selection.barrier.members.clear();
      break;
    case 1:
      selection.barrier.members.pop_back();
      break;
    case 2:
      selection.barrier.members.push_back(selection.barrier.members.front());
      break;
    case 3:
      selection.barrier.members.back() = selection.barrier.members.front();
      break;
    case 4:
      selection.barrier.members.back().bytes[0] ^= 1;
      break;
    case 5:
      selection.owner.entry = "kernel";
      break;
    case 6:
      ++selection.owner.entry_offset;
      break;
    case 7:
      ++selection.owner.descriptor_file_offset;
      break;
    case 8:
      selection = DropBarrierSelection{};
      break;
    case 9:
      ++selection.barrier.block_offset;
      break;
    case 10:
      --selection.barrier.barrier_id;
      break;
    case 11:
      selection.barrier.scope = StaticBarrierScope::Cluster;
      break;
    case 12:
      selection.kind = static_cast<FaultMutationKind>(99);
      break;
    }
    EXPECT_TRUE(apply_drop_barrier(context, selection, error).failed());
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
    const FaultMutationContext context(image);
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
    EXPECT_NE(error.code, FaultMutationErrorCode::None);
  }
}

TEST(FaultDropBarrier, SameBlockStaticPairInCycleAffectsAllOccurrences) {
  const std::vector<uint8_t> image = fixture({kSignal, kMov, kWait, 0xbfa1fffcu, kEnd});
  const FaultMutationContext context(image);
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers.size(), 1u);
  FailureOr<DropBarrierResult> result =
      apply_drop_barrier(context, selection_from(inventory.value()), error);
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
    EXPECT_TRUE(discover_drop_barriers(FaultMutationContext(fixture(words)), "test_kernel", error)
                    .failed());
  }
}

TEST(FaultDropBarrier, RejectsForeignHeadersAndNonexactEntry) {
  const std::vector<uint8_t> original = fixture();
  for (const size_t index :
       {size_t{EI_CLASS}, size_t{EI_DATA}, size_t{EI_OSABI}, size_t{EI_ABIVERSION}}) {
    std::vector<uint8_t> image = original;
    image[index] = 0;
    const FaultMutationContext context(image);
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
  }
  const FaultMutationContext context(original);
  FaultMutationError error;
  EXPECT_TRUE(discover_drop_barriers(context, "kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::BadIdentity);
  auto foreign = original;
  Elf64_Ehdr header{};
  std::memcpy(&header, foreign.data(), sizeof(header));
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX90A;
  std::memcpy(foreign.data(), &header, sizeof(header));
  const FaultMutationContext wrong_target(foreign);
  EXPECT_TRUE(discover_drop_barriers(wrong_target, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedTarget);
}

TEST(FaultDropBarrier, IndependentValidatorRejectsHalfDropAndUnrelatedCorruption) {
  const std::vector<uint8_t> image = fixture();
  const FaultMutationContext context(image);
  FaultMutationError error;
  FailureOr<FaultBarrierInventory> inventory =
      discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const DropBarrierSelection selection = selection_from(inventory.value());
  FailureOr<DropBarrierResult> result = apply_drop_barrier(context, selection, error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  for (const size_t offset :
       {size_t{0x104}, size_t{0x10c}, size_t{0x108}, size_t{0x110}, size_t{0x118}, size_t{48}}) {
    std::vector<uint8_t> candidate = result.value().edit_result.image;
    candidate[offset] ^= 1;
    EXPECT_TRUE(validate_drop_barrier_candidate(context, selection, candidate, error).failed())
        << offset;
  }
  EXPECT_TRUE(validate_drop_barrier_candidate(context, selection, image, error).failed());
}

TEST(FaultDropBarrier, CapabilityDescribesTransformationOnly) {
  const auto capabilities = fault_mutation_capabilities();
  ASSERT_EQ(capabilities.size(), std::size(kDropRows));
  for (const DropTargetRow &row : kDropRows) {
    const auto match = std::ranges::find_if(capabilities, [&](const auto &capability) {
      return capability.target == row.target && capability.wave_size == row.wave_size;
    });
    ASSERT_NE(match, capabilities.end()) << row.name;
    EXPECT_EQ(match->architecture, row.arch);
    EXPECT_EQ(match->form,
              row.split ? StaticBarrierForm::BuiltinSplitPair : StaticBarrierForm::FullSingleton);
    EXPECT_FALSE(match->admission.empty());
  }
  constexpr std::array<IsaTargetDescriptor, 0> providers{};
  const IsaTargetRegistry missing(providers);
  EXPECT_TRUE(fault_mutation_capabilities(missing).empty());
  EXPECT_FALSE(static_barrier_policy(kTarget, missing));
  // The actual standalone target is linked to model-only providers.
  EXPECT_FALSE(
      default_isa_target_registry().find_gpu_target(kTarget)->capabilities.execution_implemented);
}

TEST(FaultDropBarrier, BuiltinIdsKeepDistinctStaticScopes) {
  for (uint32_t selector = 193; selector <= 196; ++selector) {
    const int32_t id = -static_cast<int32_t>(selector - 192);
    const std::vector<uint8_t> image =
        fixture({0xbe804e00u | selector, 0xbf940000u | static_cast<uint16_t>(id), kEnd});
    const FaultMutationContext context(image);
    FaultMutationError error;
    FailureOr<FaultBarrierInventory> inventory =
        discover_drop_barriers(context, "test_kernel", error);
    ASSERT_TRUE(inventory.succeeded()) << error.message;
    ASSERT_EQ(inventory.value().barriers.size(), 1u);
    EXPECT_EQ(inventory.value().barriers.front().barrier_id, id);
    EXPECT_EQ(inventory.value().barriers.front().scope,
              id >= -2 ? StaticBarrierScope::Workgroup : StaticBarrierScope::Cluster);
    EXPECT_TRUE(apply_drop_barrier(context, selection_from(inventory.value()), error).succeeded())
        << error.message;
  }
}

TEST(FaultDropBarrier, RejectsAliasedEntryOwnersAndMalformedOmittedDescriptor) {
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
    const FaultMutationContext context(image);
    FaultMutationError error;
    EXPECT_TRUE(discover_drop_barriers(context, "test_kernel0", error).failed());
    EXPECT_EQ(error.code, malformed ? FaultMutationErrorCode::UnsupportedForm
                                    : FaultMutationErrorCode::AmbiguousSelection)
        << error.message;
  }
}

TEST(FaultDropBarrier, Gfx1250WaveModeComesFromTargetDespiteMissingLegacyBit) {
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
  const FaultMutationContext context(image);
  FaultMutationError error;
  const auto inventory = discover_drop_barriers(context, "test_kernel", error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  const auto result = apply_drop_barrier(context, selection_from(inventory.value()), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(std::memcmp(result.value().edit_result.image.data() +
                            descriptors.front().descriptor_file_offset,
                        image.data() + descriptors.front().descriptor_file_offset,
                        sizeof(descriptor)),
            0);
}

TEST(FaultDropBarrier, RejectsBarrierInsideClause) {
  FaultMutationError error;
  EXPECT_TRUE(
      discover_drop_barriers(FaultMutationContext(fixture({0xbf850001u, kSignal, kWait, kEnd})),
                             "test_kernel", error)
          .failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
}

TEST(FaultDropBarrier, RejectsTrapBetweenSignalAndWait) {
  FaultMutationError error;
  const std::vector<uint8_t> image = fixture({kSignal, 0xbf900001u, kWait, kEnd});
  const FaultMutationContext context(image);
  EXPECT_TRUE(discover_drop_barriers(context, "test_kernel", error).failed());
  EXPECT_EQ(error.code, FaultMutationErrorCode::UnsupportedForm);
}

#ifdef RJ_FAULT_CORPUS_OBJECT
TEST(FaultDropBarrier, PinnedCompiledCorpusCompleteDrop) {
  std::ifstream input(RJ_FAULT_CORPUS_OBJECT, std::ios::binary);
  ASSERT_TRUE(input.is_open());
  const std::vector<uint8_t> image((std::istreambuf_iterator<char>(input)), {});
  ASSERT_FALSE(input.bad());
  const FaultMutationContext context(image);
  FaultMutationError error;
  constexpr std::string_view entry = "_Z16fa_barrier_epochPKtPjj";
  FailureOr<FaultBarrierInventory> inventory = discover_drop_barriers(context, entry, error);
  ASSERT_TRUE(inventory.succeeded()) << error.message;
  ASSERT_EQ(inventory.value().barriers.size(), 1u);
  const StaticBarrier &group = inventory.value().barriers.front();
  EXPECT_EQ(group.barrier_id, -1);
  EXPECT_EQ(group.scope, StaticBarrierScope::Workgroup);
  ASSERT_EQ(group.members.size(), 2u);
  EXPECT_GT(group.members[1].offset, group.members[0].offset + group.members[0].bytes.size());
  FailureOr<DropBarrierResult> result =
      apply_drop_barrier(context, selection_from(inventory.value()), error);
  ASSERT_TRUE(result.succeeded()) << error.message;
  EXPECT_EQ(result.value().edit_result.edits.size(), 2u);
  EXPECT_TRUE(validate_drop_barrier_candidate(context, result.value().selection,
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
