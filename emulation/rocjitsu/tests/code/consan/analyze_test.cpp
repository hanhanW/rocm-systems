// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "capture_fixture.h"
#include "rocjitsu/code/consan/analysis/analyze.h"
#include "rocjitsu/code/consan/encoding.h"

#include <gtest/gtest.h>

#include <algorithm>

namespace rocjitsu::consan {
namespace {

class ConSanAnalysis : public ::testing::Test {
protected:
  void SetUp() override {
    capture_.header = test::capture_header();
    capture_.header.manifest.entry.original_group_bytes = 128;
    capture_.header.manifest.entry.prepared_group_bytes = 128;
    append_execution();
  }

  void append_execution(LaunchDimensions dimensions = {{64, 1, 1}, {64, 1, 1}}) {
    const uint64_t serial = capture_.executions.size() + 1;
    CapturedExecution execution =
        test::execution(capture_.header.manifest, serial, serial, dimensions);
    const FailureOr<LaunchLayout> layout =
        plan_launch(capture_.header.manifest, execution.dimensions, execution.group_bytes, 0);
    ASSERT_TRUE(layout.succeeded());
    stride_ = layout.value().row_stride;
    for (uint32_t wave = 0; wave < layout.value().wave_count; ++wave)
      wire::write_u32(execution.report, row(wave) + wire::kRowStatus,
                      wire::kRowEntered | wire::kRowExited);
    capture_.executions.push_back(std::move(execution));
  }

  size_t row(uint32_t wave) const { return kReportHeaderBytes + wave * stride_; }
  size_t record(uint32_t wave, uint32_t site, uint32_t address_group = 0) const {
    return row(wave) + kRowHeaderBytes +
           (site * kAddressGroupsPerSite + address_group) * kAccessRecordBytes;
  }

  void access(uint32_t wave, uint32_t site, uint32_t sequence, uint32_t offset = 0,
              uint64_t mask = 1, uint32_t address_group = 0, uint32_t execution = 0) {
    std::vector<uint8_t> &report = capture_.executions[execution].report;
    const SiteFacts &facts = capture_.header.manifest.sites[site];
    const size_t base = record(wave, site, address_group);
    for (const auto &[field, value] : std::array<std::pair<uint32_t, uint32_t>, 9>{
             {{wire::kRecordCommit, wire::kAccessCommitted},
              {wire::kRecordAnalysis, uint32_t(facts.analysis)},
              {wire::kRecordKind, uint32_t(facts.record)},
              {wire::kRecordSite, site},
              {wire::kRecordRange, facts.range_id},
              {wire::kRecordSequence, sequence},
              {wire::kRecordLdsOffset, offset},
              {wire::kRecordWidth, facts.width},
              {wire::kRecordFlags, wire::kExactGuestMask}}})
      wire::write_u32(report, base + field, value);
    wire::write_u64(report, base + wire::kRecordGuestMask, mask);
    const uint32_t final = *wire::read_u32(report, row(wave) + wire::kRowFinalSequence);
    wire::write_u32(report, row(wave) + wire::kRowFinalSequence, std::max(final, sequence));
  }

  FailureOr<std::vector<uint8_t>> encode() {
    CaptureEncoder encoder;
    FailureOr<std::vector<uint8_t>> bytes = encoder.start(capture_.header, &error_);
    if (bytes.failed())
      return Result::failure();
    for (const CapturedExecution &execution : capture_.executions) {
      const FailureOr<std::vector<uint8_t>> frame = encoder.append(execution, &error_);
      if (frame.failed())
        return Result::failure();
      bytes.value().insert(bytes.value().end(), frame.value().begin(), frame.value().end());
    }
    const FailureOr<std::vector<uint8_t>> end = encoder.finish(&error_);
    if (end.failed())
      return Result::failure();
    bytes.value().insert(bytes.value().end(), end.value().begin(), end.value().end());
    return bytes;
  }

  FailureOr<LdsRaceResult> analyze(AnalysisOptions options = {}) {
    const FailureOr<std::vector<uint8_t>> bytes = encode();
    if (bytes.failed())
      return Result::failure();
    const FailureOr<Capture> decoded = decode_capture(bytes.value(), &error_);
    if (decoded.failed())
      return Result::failure();
    return analyze_capture(decoded.value(), options, &error_);
  }

  void many_sites(uint32_t count) {
    Manifest &facts = capture_.header.manifest;
    facts.sites.clear();
    facts.original_text_bytes = 8 * count + 64;
    facts.prepared_text_bytes = 128 * count + 768;
    facts.entry.original_descriptor_offset = facts.original_text_bytes + 64;
    facts.entry.prepared_descriptor_offset = facts.prepared_text_bytes + 64;
    facts.entry.prepared_entry_offset = 128 * count + 256;
    facts.original.bytes = facts.entry.original_descriptor_offset + 64;
    facts.prepared.bytes = facts.entry.prepared_descriptor_offset + 64;
    for (uint32_t site = 0; site < count; ++site) {
      facts.sites.push_back({.id = site,
                             .analysis = AnalysisKind::LdsRaceDetection,
                             .access = AccessKind::Write,
                             .original_offset = uint64_t{site} * 8,
                             .original_size = 8,
                             .prepared_guest_offset = uint64_t{site} * 128,
                             .prepared_guest_size = 8,
                             .probe_offset = uint64_t{site} * 128 + 64,
                             .probe_size = 64,
                             .first_slot = site * kAddressGroupsPerSite});
    }
    capture_.executions.clear();
    append_execution();
  }

  void barrier() {
    Manifest &facts = capture_.header.manifest;
    facts.target = 1250;
    facts.scope = ScopeKind::NativeLdsWorkgroupBarrier;
    facts.sites[1].original_offset = 32;
    facts.barrier = BarrierFacts{.original_drain = 8,
                                 .original_signal = 12,
                                 .original_wait = 16,
                                 .prepared_signal = 320,
                                 .prepared_wait = 324,
                                 .completion_probe = 328,
                                 .probe_size = 64};
    std::vector<uint8_t> &report = capture_.executions.front().report;
    for (const uint32_t wave : {0u, 1u}) {
      wire::write_u32(report, row(wave) + wire::kRowBarrierCommit, wire::kBarrierCompleted);
      wire::write_u32(report, row(wave) + wire::kRowBarrierGroup, 0);
      wire::write_u32(report, row(wave) + wire::kRowBarrierOccurrence, 1);
      wire::write_u32(report, row(wave) + wire::kRowBarrierParticipants, 2);
      wire::write_u32(report, row(wave) + wire::kRowBarrierSequence, wave == 0 ? 1 : 0);
    }
  }

  void many_barrier_sites(uint32_t count, uint32_t split, uint32_t waves = 2) {
    capture_.header.manifest.barrier.reset();
    capture_.header.manifest.scope = ScopeKind::NativeLdsNoSynchronization;
    many_sites(count);
    Manifest &facts = capture_.header.manifest;
    facts.target = 1250;
    facts.scope = ScopeKind::NativeLdsWorkgroupBarrier;
    for (uint32_t site = split; site < count; ++site) {
      facts.sites[site].original_offset += 16;
      facts.sites[site].prepared_guest_offset += 128;
      facts.sites[site].probe_offset += 128;
    }
    facts.barrier = BarrierFacts{.original_drain = uint64_t{split} * 8,
                                 .original_signal = uint64_t{split} * 8 + 4,
                                 .original_wait = uint64_t{split} * 8 + 8,
                                 .prepared_signal = uint64_t{split} * 128,
                                 .prepared_wait = uint64_t{split} * 128 + 4,
                                 .completion_probe = uint64_t{split} * 128 + 8,
                                 .probe_size = 64};
    capture_.executions.clear();
    append_execution({{32 * waves, 1, 1}, {32 * waves, 1, 1}});
    std::vector<uint8_t> &report = capture_.executions.front().report;
    for (uint32_t wave = 0; wave < waves; ++wave) {
      wire::write_u32(report, row(wave) + wire::kRowBarrierCommit, wire::kBarrierCompleted);
      wire::write_u32(report, row(wave) + wire::kRowBarrierOccurrence, 1);
      wire::write_u32(report, row(wave) + wire::kRowBarrierParticipants, waves);
      wire::write_u32(report, row(wave) + wire::kRowBarrierSequence, split);
    }
  }

  Capture capture_;
  ContractError error_;
  uint32_t stride_ = 0;
};

TEST_F(ConSanAnalysis, OverlappingWriteReadProducesAttributedUnorderedCrossWaveFinding) {
  access(0, 0, 1);
  access(1, 1, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_TRUE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 1u);
  EXPECT_EQ(result.value().collision_groups, 0u);
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.valid_observations, 2u);
  EXPECT_EQ(execution.cross_wave_lane_pairs, 1u);
  ASSERT_EQ(execution.cross_wave_examples.size(), 1u);
  const LdsRaceFinding &finding = execution.cross_wave_examples.front();
  EXPECT_EQ(finding.first.wave, 0u);
  EXPECT_EQ(finding.first.site, 0u);
  EXPECT_EQ(finding.first.original_offset, 0u);
  EXPECT_EQ(finding.second.wave, 1u);
  EXPECT_EQ(finding.second.site, 1u);
  EXPECT_EQ(finding.second.original_offset, 8u);
  EXPECT_EQ(finding.first.mask, 1u);
  EXPECT_FALSE(execution.ordering_uncertain);
}

TEST_F(ConSanAnalysis, FixedBarrierCertificateOrdersRetainedPairAndSurvivesUnrelatedLocalGap) {
  access(0, 0, 1);
  access(1, 1, 1);
  barrier();
  for (const bool unrelated_gap : {false, true}) {
    if (unrelated_gap)
      wire::write_u32(capture_.executions[0].report, record(0, 1) + wire::kRecordCommit, 7);
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message;
    const LdsRaceExecutionResult &execution = result.value().executions.front();
    EXPECT_EQ(execution.valid_observations, 2u);
    EXPECT_EQ(execution.ordered_pairs, 1u);
    EXPECT_EQ(execution.cross_wave_pairs, 0u);
    EXPECT_EQ(execution.ordering_unresolved_pairs, 0u);
    EXPECT_EQ(execution.completed_barrier_waves, 2u);
    EXPECT_EQ(execution.coverage_complete, !unrelated_gap);
    ASSERT_EQ(execution.ordered_examples.size(), 1u);
    EXPECT_EQ(execution.ordered_examples[0].ordering, LdsRaceOrdering::Ordered);
    EXPECT_TRUE(execution.ordered_examples[0].first.mask == 1);
  }
}

TEST_F(ConSanAnalysis,
       DecoderControlMissingParticipantOccurrenceOrCompletionRetainsUnresolvedPair) {
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  barrier();
  const std::vector<uint8_t> saved = capture_.executions[0].report;
  for (const uint32_t field :
       {wire::kRowBarrierCommit, wire::kRowBarrierGroup, wire::kRowBarrierOccurrence,
        wire::kRowBarrierParticipants, wire::kRowBarrierSequence}) {
    SCOPED_TRACE(field);
    capture_.executions[0].report = saved;
    wire::write_u32(capture_.executions[0].report, row(1) + field, 9);
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message;
    const LdsRaceExecutionResult &execution = result.value().executions.front();
    EXPECT_EQ(execution.ordered_pairs, 0u);
    EXPECT_EQ(execution.cross_wave_pairs, 1u);
    EXPECT_EQ(execution.ordering_unresolved_pairs, 1u);
    EXPECT_EQ(execution.collision_groups, 1u);
    EXPECT_FALSE(execution.coverage_complete);
    ASSERT_EQ(execution.cross_wave_examples.size(), 1u);
    EXPECT_EQ(execution.cross_wave_examples[0].ordering, LdsRaceOrdering::Incomplete);
    EXPECT_NE(render_lds_race_analysis(result.value()).find("ordering_unresolved"),
              std::string::npos);
  }
}

TEST_F(ConSanAnalysis, FixedBarrierDoesNotEraseIndependentSameInstructionCollision) {
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  barrier();
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.ordered_pairs, 1u);
  EXPECT_EQ(execution.cross_wave_pairs, 0u);
  EXPECT_EQ(execution.collision_groups, 1u);
  EXPECT_TRUE(execution.coverage_complete);
}

TEST_F(ConSanAnalysis, FixedBarrierReadReadAndDisjointKeepObservationsWithoutCandidates) {
  for (const bool read_read : {false, true}) {
    capture_.executions.clear();
    append_execution();
    capture_.header.manifest.sites[0].access = read_read ? AccessKind::Read : AccessKind::Write;
    access(0, 0, 1);
    access(1, 1, 1, read_read ? 0 : 4);
    barrier();
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message;
    const LdsRaceExecutionResult &execution = result.value().executions.front();
    EXPECT_EQ(execution.valid_observations, 2u);
    EXPECT_EQ(execution.ordered_pairs, 0u);
    EXPECT_EQ(execution.cross_wave_pairs, 0u);
    EXPECT_EQ(execution.collision_groups, 0u);
    EXPECT_TRUE(execution.coverage_complete);
  }
}

TEST_F(ConSanAnalysis, DecoderControlCompletionSequenceMustBracketTheActualEndpoints) {
  access(0, 0, 1);
  access(1, 1, 1);
  barrier();
  // This otherwise valid completion wrongly places the consumer before wait.
  wire::write_u32(capture_.executions[0].report, row(1) + wire::kRowBarrierSequence, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.completed_barrier_waves, 2u);
  EXPECT_EQ(execution.ordered_pairs, 0u);
  EXPECT_EQ(execution.ordering_unresolved_pairs, 1u);
}

TEST_F(ConSanAnalysis, DecoderControlEmptyAccessWaveStillMustCompleteTheSameBarrier) {
  capture_.executions.clear();
  append_execution({{65, 1, 1}, {65, 1, 1}});
  access(0, 0, 1);
  access(1, 1, 1);
  barrier();
  for (const uint32_t wave : {0u, 1u})
    wire::write_u32(capture_.executions[0].report, row(wave) + wire::kRowBarrierParticipants, 3);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.valid_observations, 2u);
  EXPECT_EQ(execution.required_barrier_waves, 3u);
  EXPECT_EQ(execution.completed_barrier_waves, 2u);
  EXPECT_EQ(execution.ordered_pairs, 0u);
  EXPECT_EQ(execution.ordering_unresolved_pairs, 1u);
}

TEST_F(ConSanAnalysis, SameSideOfBarrierRemainsUnorderedWhenCertificateIsMissing) {
  access(0, 0, 1);
  access(1, 1, 1);
  barrier();
  capture_.header.manifest.sites[1].original_offset = 64;
  capture_.header.manifest.barrier->original_drain = 72;
  capture_.header.manifest.barrier->original_signal = 76;
  capture_.header.manifest.barrier->original_wait = 80;
  wire::write_u32(capture_.executions[0].report, row(1) + wire::kRowBarrierCommit, 0);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.cross_wave_pairs, 1u);
  EXPECT_EQ(execution.ordering_unresolved_pairs, 0u);
  EXPECT_EQ(execution.cross_wave_examples[0].ordering, LdsRaceOrdering::Unordered);
}

TEST_F(ConSanAnalysis, AdjacentEffectiveByteRangesAreDisjointWithNonemptyObservations) {
  access(0, 0, 1, 0);
  access(1, 1, 1, 4);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().executions.front().valid_observations, 2u);
  EXPECT_TRUE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, OverlappingReadReadGroupsAreNotConflictCandidates) {
  capture_.header.manifest.sites[0].access = AccessKind::Read;
  access(0, 0, 1, 0, 3);
  access(1, 1, 1, 0, UINT32_MAX);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().collision_groups, 0u);
  EXPECT_EQ(result.value().executions.front().valid_observations, 2u);
}

TEST_F(ConSanAnalysis, DifferentInstructionsInOneWaveDoNotBecomeCrossWavePairs) {
  access(0, 0, 1);
  access(0, 1, 2);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().collision_groups, 0u);
  EXPECT_TRUE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, ReusedStorageAcrossDispatchesCannotJoinOtherwiseOverlappingGroups) {
  append_execution();
  access(0, 0, 1, 0, 1, 0, 0);
  access(1, 1, 1, 0, 1, 0, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  ASSERT_EQ(result.value().executions.size(), 2u);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().executions[0].valid_observations, 1u);
  EXPECT_EQ(result.value().executions[1].valid_observations, 1u);
  EXPECT_EQ(result.value().executions[1].serial, 2u);
}

TEST_F(ConSanAnalysis, ExactOrdinaryWriteMaskReportsFullParticipatingLanesAndPairCount) {
  access(0, 0, 1, 0, 0x80000005u);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_EQ(result.value().collision_groups, 1u);
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.collision_lane_pairs, 3u);
  ASSERT_EQ(execution.collision_examples.size(), 1u);
  EXPECT_EQ(execution.collision_examples.front().first.mask, 0x80000005u);
  EXPECT_EQ(execution.valid_observations, 1u);
  EXPECT_EQ(render_analysis(result.value()).find("\"ordering\":"), std::string::npos);
  EXPECT_NE(render_analysis(result.value()).find("ordinary_store_exact_mask"), std::string::npos);
}

TEST_F(ConSanAnalysis, SingleLaneWriteAndMultiLaneReadNeverInventInstructionCollisions) {
  access(0, 0, 1, 0, 0x80000000u);
  access(0, 1, 2, 0, UINT32_MAX);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_EQ(result.value().collision_groups, 0u);
  EXPECT_EQ(result.value().executions.front().valid_observations, 2u);
}

TEST_F(ConSanAnalysis, BothChecksConsumeTheSameWriteObservation) {
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.valid_observations, 2u);
  EXPECT_EQ(execution.cross_wave_pairs, 1u);
  EXPECT_EQ(execution.cross_wave_lane_pairs, 2u);
  EXPECT_EQ(execution.collision_groups, 1u);
  EXPECT_EQ(execution.cross_wave_examples.front().first,
            execution.collision_examples.front().first);
}

TEST_F(ConSanAnalysis, ExampleLimitsNeverChangeCompleteGroupOrLanePairCounts) {
  capture_.header.manifest.sites[1].access = AccessKind::Write;
  for (uint32_t wave = 0; wave < 2; ++wave) {
    access(wave, 0, 1, 0, 3);
    access(wave, 1, 2, 0, 3);
  }
  for (const uint32_t limit : {0, 1, 3, 8}) {
    const FailureOr<LdsRaceResult> result = analyze({limit});
    ASSERT_TRUE(result.succeeded()) << error_.message;
    const LdsRaceExecutionResult &execution = result.value().executions.front();
    EXPECT_EQ(execution.cross_wave_pairs, 4u);
    EXPECT_EQ(execution.cross_wave_lane_pairs, 16u);
    EXPECT_EQ(execution.collision_groups, 4u);
    EXPECT_EQ(execution.collision_lane_pairs, 4u);
    EXPECT_EQ(execution.cross_wave_examples.size(), std::min(limit, 4u));
    EXPECT_EQ(execution.collision_examples.size(), std::min(limit, 4u));
  }
}

TEST_F(ConSanAnalysis, MalformedRowDoesNotEraseAnotherWavesIndependentCollision) {
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  wire::write_u32(capture_.executions[0].report, row(1) + wire::kRowStatus, UINT32_MAX);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().collision_groups, 1u);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_TRUE(result.value().executions.front().ordering_uncertain);
}

TEST_F(ConSanAnalysis, MalformedLocalRecordPreservesAnIndependentCrossWavePair) {
  access(0, 0, 1);
  access(0, 1, 2);
  access(1, 1, 1);
  wire::write_u32(capture_.executions[0].report, record(0, 1) + wire::kRecordCommit, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 1u);
  EXPECT_EQ(result.value().executions.front().rejected_records, 1u);
}

TEST_F(ConSanAnalysis, InvalidRecordIdentityMaskAndBoundsAreGapsBeyondTheContainerChecksum) {
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  std::vector<uint8_t> &report = capture_.executions.front().report;
  const std::vector<uint8_t> saved = report;
  for (const auto &[field, value] :
       std::array<std::pair<uint32_t, uint32_t>, 12>{{{wire::kRecordCommit, 1},
                                                      {wire::kRecordAnalysis, 2},
                                                      {wire::kRecordKind, 2},
                                                      {wire::kRecordSite, 99},
                                                      {wire::kRecordRange, 1},
                                                      {wire::kRecordSequence, 2},
                                                      {wire::kRecordWidth, 8},
                                                      {wire::kRecordGuestMask, 0},
                                                      {wire::kRecordFlags, 0},
                                                      {wire::kRecordFlags, 3},
                                                      {wire::kRecordLdsOffset, 128},
                                                      {wire::kRecordReservedBegin, 1}}}) {
    report = saved;
    wire::write_u32(report, record(1, 1) + field, value);
    // Production encoder makes a valid whole container and frame checksum.
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message;
    EXPECT_FALSE(result.value().coverage_complete) << field;
    EXPECT_EQ(result.value().cross_wave_pairs, 0u) << field;
    EXPECT_EQ(result.value().collision_groups, 1u) << field;
  }
}

TEST_F(ConSanAnalysis, MissingStaticMappingIsAnInputErrorNotInstructionZero) {
  access(0, 0, 1);
  capture_.header.manifest.sites[0].original_size = 0;
  EXPECT_TRUE(analyze_capture(capture_, {}, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
}

TEST_F(ConSanAnalysis, UnmodeledScopeCannotBeTreatedAsConfirmedUnorderedAccesses) {
  access(0, 0, 1);
  access(1, 1, 1);
  capture_.header.manifest.scope = static_cast<ScopeKind>(2);
  EXPECT_TRUE(analyze_capture(capture_, {}, &error_).failed());
  EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
}

TEST_F(ConSanAnalysis, TypedHeaderConditionsAndIdentityCannotBypassFileValidation) {
  const CaptureHeader saved = capture_.header;
  for (const uint32_t unsupported : {0u, 14u, 13u, 11u, 7u, 31u}) {
    capture_.header.execution_conditions = unsupported;
    EXPECT_TRUE(analyze_capture(capture_, {}, &error_).failed());
    EXPECT_EQ(error_.message, "unsupported capture execution conditions");
  }
  capture_.header = saved;
  capture_.header.run_identity = {};
  EXPECT_TRUE(analyze_capture(capture_, {}, &error_).failed());
  EXPECT_EQ(error_.message, "missing capture run identity");
  capture_.header = saved;
  EXPECT_TRUE(analyze_capture(capture_).succeeded());
}

TEST_F(ConSanAnalysis, TypedAnalysisBudgetsRetainTheirIndependentLimits) {
  EXPECT_TRUE(analyze_capture(capture_, {kMaxFindingExamples}).succeeded());
  EXPECT_TRUE(analyze_capture(capture_, {kMaxFindingExamples + 1}, &error_).failed());
  EXPECT_EQ(error_.message, "finding example budget exceeded");
  capture_.executions.resize(CaptureLimits{}.executions + 1);
  EXPECT_TRUE(analyze_capture(capture_, {}, &error_).failed());
  EXPECT_EQ(error_.message, "capture execution budget exceeded");
}

TEST_F(ConSanAnalysis, InvalidReportParentPreservesFindingsFromAnotherExecution) {
  append_execution();
  access(0, 0, 1, 0, 1, 0, 1);
  access(1, 1, 1, 0, 1, 0, 1);
  wire::write_u32(capture_.executions[0].report, wire::kReportRowStride, UINT32_MAX);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 1u);
  EXPECT_EQ(result.value().executions[0].valid_observations, 0u);
  EXPECT_EQ(result.value().executions[1].cross_wave_pairs, 1u);
}

TEST_F(ConSanAnalysis, RowIdentityPublicationAndFinalSequenceMustMatchTheirExecution) {
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  std::vector<uint8_t> &report = capture_.executions.front().report;
  const std::vector<uint8_t> saved = report;
  for (const auto &[field, value] :
       std::array<std::pair<uint32_t, uint32_t>, 8>{{{wire::kRowSerial, 2},
                                                     {wire::kRowEntry, 1},
                                                     {wire::kRowWave, 0},
                                                     {wire::kRowStatus, wire::kRowEntered},
                                                     {wire::kRowLoss, 1},
                                                     {wire::kRowFinalSequence, 0},
                                                     {wire::kRowFinalSequence, UINT32_MAX},
                                                     {wire::kRowReservedBegin, 1}}}) {
    report = saved;
    wire::write_u32(report, row(1) + field, value);
    const FailureOr<LdsRaceResult> result = analyze();
    ASSERT_TRUE(result.succeeded()) << error_.message;
    EXPECT_FALSE(result.value().coverage_complete);
    EXPECT_EQ(result.value().cross_wave_pairs, 0u);
    EXPECT_EQ(result.value().collision_groups, 1u);
  }
}

TEST_F(ConSanAnalysis, InvertedSiteOccurrenceOrderCannotBeAValidForwardOnlyExecution) {
  access(0, 0, 2);
  access(0, 1, 1);
  access(1, 1, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
}

TEST_F(ConSanAnalysis, ShortTypedReportCannotEscapeTheCoreBoundsCheck) {
  capture_.executions[0].report.resize(1);
  const FailureOr<LdsRaceResult> result = analyze_capture(capture_, {}, &error_);
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().executions.front().valid_observations, 0u);
}

TEST_F(ConSanAnalysis, DuplicateOccurrencesAndOverlappingMasksCannotInflateFindings) {
  access(0, 0, 1);
  access(0, 1, 1);
  access(1, 1, 1);
  FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().executions[0].rejected_records, 2u);
  // Another address group in the same instruction cannot reuse lane zero.
  std::ranges::fill(
      std::span(capture_.executions[0].report).subspan(record(0, 1), kAccessRecordBytes), 0);
  access(0, 0, 1, 4, 1, 1);
  result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_FALSE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, ReservationLossRetainsValidFindingsButMakesCoverageIncomplete) {
  for (uint32_t address_group = 0; address_group < 4; ++address_group)
    access(0, 0, 1, address_group * 4, 1u << address_group, address_group);
  access(1, 1, 1);
  wire::write_u32(capture_.executions[0].report, row(0) + wire::kRowLoss, 1);
  wire::write_u32(capture_.executions[0].report, row(0) + wire::kRowStatus,
                  wire::kRowEntered | wire::kRowExited | wire::kRowExhausted);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 1u);
  EXPECT_EQ(result.value().executions.front().lost_occurrences, 1u);
  EXPECT_EQ(result.value().executions.front().valid_observations, 5u);
}

TEST_F(ConSanAnalysis, UnalignedOperandsNeedAlignmentModeEvidenceAndCannotProduceACleanVerdict) {
  access(0, 0, 1, 1, 3);
  access(1, 1, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_FALSE(result.value().coverage_complete);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().collision_groups, 0u);
  EXPECT_EQ(result.value().executions.front().valid_observations, 1u);
  EXPECT_EQ(result.value().executions.front().issues.front().reason,
            "unaligned_lds_alignment_mode_unavailable");
}

TEST_F(ConSanAnalysis, ReaderOwnedEvidenceAndRenderedSemanticReplayAreDeterministic) {
  capture_.header.manifest.entry.name = "kernel\"\\\n";
  access(0, 0, 1, 0, 3);
  access(1, 1, 1);
  FailureOr<std::vector<uint8_t>> bytes = encode();
  ASSERT_TRUE(bytes.succeeded());
  FailureOr<Capture> first = decode_capture(bytes.value());
  FailureOr<Capture> second = decode_capture(bytes.value());
  ASSERT_TRUE(first.succeeded());
  ASSERT_TRUE(second.succeeded());
  std::ranges::fill(bytes.value(), 0xcd);
  const FailureOr<LdsRaceResult> a = analyze_capture(first.value());
  const FailureOr<LdsRaceResult> b = analyze_capture(second.value());
  ASSERT_TRUE(a.succeeded());
  ASSERT_TRUE(b.succeeded());
  EXPECT_EQ(a.value(), b.value());
  EXPECT_EQ(render_analysis(a.value()), render_analysis(b.value()));
  EXPECT_NE(render_analysis(a.value()).find("kernel\\\"\\\\\\u000a"), std::string::npos);
}

TEST_F(ConSanAnalysis, AcceptedUtf8EntryNamesRetainTheirBytesInJson) {
  // Exercise two-, three- and four-byte UTF-8 alongside JSON escapes.
  const std::string name = "lds_producer_consum\xc3\xa9_\xe6\xa0\xb8_\xf0\x9f\x98\x80";
  capture_.header.manifest.entry.name = name + "\"\\\n\t";
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_EQ(result.value().entry, capture_.header.manifest.entry.name);
  EXPECT_NE(
      render_analysis(result.value()).find("\"entry\":\"" + name + "\\\"\\\\\\u000a\\u0009\""),
      std::string::npos);
}

TEST_F(ConSanAnalysis, InvalidUtf8EntryNamesAreAnalysisInputErrorsAfterBinaryRoundTrip) {
  for (const std::string_view invalid :
       {"\x80", "\xbf", "\xc0\x80", "\xc1\xbf", "\xc2", "\xc2\x7f", "\xdf\xff", "\xe0\x80\x80",
        "\xed\xa0\x80", "\xef\xbf", "\xf0\x80\x80\x80", "\xf4\x90\x80\x80", "\xf5\x80\x80\x80",
        "\xf8\x88\x80\x80\x80", "\xff"}) {
    capture_.header.manifest.entry.name = "kernel_" + std::string(invalid);
    // Binary capture remains byte preserving; only JSON-producing analysis
    // requires text. Re-encode each case so checksums cannot hide this check.
    const FailureOr<std::vector<uint8_t>> bytes = encode();
    ASSERT_TRUE(bytes.succeeded()) << error_.message;
    const FailureOr<Capture> decoded = decode_capture(bytes.value(), &error_);
    ASSERT_TRUE(decoded.succeeded()) << error_.message;
    EXPECT_EQ(decoded.value().header.manifest.entry.name, capture_.header.manifest.entry.name);
    const FailureOr<LdsRaceResult> result = analyze_capture(decoded.value(), {}, &error_);
    EXPECT_TRUE(result.failed());
    EXPECT_EQ(error_.code, ContractErrorCode::InvalidFacts);
    EXPECT_EQ(error_.message, "entry name is not valid UTF-8 for JSON output");
  }
}

TEST_F(ConSanAnalysis, Utf8ScalarBoundariesRetainTheirBytesInJson) {
  const std::string name = "kernel_\xc2\x80\xdf\xbf\xe0\xa0\x80\xed\x9f\xbf"
                           "\xee\x80\x80\xef\xbf\xbf\xf0\x90\x80\x80\xf4\x8f\xbf\xbf";
  capture_.header.manifest.entry.name = name;
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_NE(render_analysis(result.value()).find("\"entry\":\"" + name + "\""), std::string::npos);
}

TEST_F(ConSanAnalysis, EmptyValidObservationsAndNoExecutionsNeverClaimRaceFreedom) {
  FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_EQ(result.value().executions.front().valid_observations, 0u);
  EXPECT_NE(render_analysis(result.value()).find("no_observed_findings"), std::string::npos);
  EXPECT_EQ(render_analysis(result.value()).find("race_free"), std::string::npos);
  capture_.executions.clear();
  result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_FALSE(result.value().coverage_complete);
}

TEST_F(ConSanAnalysis, LargeCandidateCountsRemainCompleteWithOneExample) {
  constexpr uint32_t sites = 1024;
  many_sites(sites);
  for (uint32_t wave = 0; wave < 2; ++wave)
    for (uint32_t site = 0; site < sites; ++site)
      access(wave, site, site + 1, 0, 3);
  const FailureOr<LdsRaceResult> result = analyze({1});
  ASSERT_TRUE(result.succeeded()) << error_.message;
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_TRUE(execution.coverage_complete);
  EXPECT_EQ(execution.cross_wave_pairs, uint64_t{sites} * sites);
  EXPECT_EQ(execution.cross_wave_lane_pairs, 4 * uint64_t{sites} * sites);
  EXPECT_EQ(execution.collision_groups, 2 * sites);
  EXPECT_EQ(execution.cross_wave_examples.size(), 1u);
  EXPECT_EQ(execution.collision_examples.size(), 1u);
}

TEST_F(ConSanAnalysis, LargeBarrierCountsRetainAllPairsWithBoundedExamples) {
  constexpr uint32_t sites = 1024;
  for (const uint32_t split : {0u, sites / 2, sites})
    for (const bool missing : {false, true}) {
      many_barrier_sites(sites, split);
      for (uint32_t wave = 0; wave < 2; ++wave)
        for (uint32_t site = 0; site < sites; ++site)
          access(wave, site, site + 1, 0, wave == 0 ? 3 : 7);
      if (missing)
        wire::write_u32(capture_.executions[0].report, row(1) + wire::kRowBarrierCommit, 0);
      for (const uint32_t examples : {0u, 1u, 7u}) {
        SCOPED_TRACE(split);
        SCOPED_TRACE(missing);
        SCOPED_TRACE(examples);
        const FailureOr<LdsRaceResult> result = analyze({examples});
        ASSERT_TRUE(result.succeeded()) << error_.message;
        const LdsRaceExecutionResult &execution = result.value().executions.front();
        const uint64_t separated = 2 * uint64_t{split} * (sites - split);
        const uint64_t ordered = missing ? 0 : separated;
        const uint64_t retained = uint64_t{sites} * sites - ordered;
        EXPECT_EQ(execution.valid_observations, 2u * sites);
        EXPECT_EQ(execution.ordered_pairs, ordered);
        EXPECT_EQ(execution.cross_wave_pairs, retained);
        EXPECT_EQ(execution.cross_wave_lane_pairs, 6 * retained);
        EXPECT_EQ(execution.ordering_unresolved_pairs, missing ? separated : 0);
        EXPECT_EQ(execution.coverage_complete, !missing);
        EXPECT_EQ(execution.collision_groups, 2u * sites);
        EXPECT_EQ(execution.ordered_examples.size(), std::min(uint64_t{examples}, ordered));
        EXPECT_EQ(execution.cross_wave_examples.size(), std::min(uint64_t{examples}, retained));
        for (const LdsRaceFinding &finding : execution.ordered_examples) {
          EXPECT_EQ(finding.ordering, LdsRaceOrdering::Ordered);
          EXPECT_NE(finding.first.site < split, finding.second.site < split);
        }
      }
    }
}

TEST_F(ConSanAnalysis, BarrierCountsAndExampleOrderMatchExhaustiveCandidatePairs) {
  constexpr uint32_t sites = 8, split = 4, waves = 4;
  const std::array<uint32_t, waves> boundaries{2, 6, 0, 8};
  for (const bool missing : {false, true}) {
    many_barrier_sites(sites, split, waves);
    for (uint32_t wave = 0; wave < waves; ++wave) {
      wire::write_u32(capture_.executions[0].report, row(wave) + wire::kRowBarrierSequence,
                      boundaries[wave]);
      for (uint32_t site = 0; site < sites; ++site)
        access(wave, site, site + 1, 0, (uint64_t{1} << (wave + 1)) - 1);
    }
    if (missing)
      wire::write_u32(capture_.executions[0].report, row(3) + wire::kRowBarrierCommit, 0);
    for (const uint32_t examples : {0u, 1u, 7u, 64u}) {
      std::vector<LdsRaceFinding> ordered, retained;
      uint64_t ordered_count = 0, unresolved_count = 0, retained_count = 0, lane_pairs = 0;
      // Brute-force the original candidate order independently of classification
      // buckets, including both static sides and contradictory dynamic boundaries.
      for (uint32_t second_wave = 1; second_wave < waves; ++second_wave)
        for (uint32_t first_wave = 0; first_wave < second_wave; ++first_wave)
          for (uint32_t first_site = 0; first_site < sites; ++first_site)
            for (uint32_t second_site = 0; second_site < sites; ++second_site) {
              LdsRaceOrdering order = LdsRaceOrdering::Unordered;
              if ((first_site < split) != (second_site < split)) {
                const bool first_valid =
                    (first_site + 1 <= boundaries[first_wave]) == (first_site < split);
                const bool second_valid =
                    (second_site + 1 <= boundaries[second_wave]) == (second_site < split);
                order = !missing && first_valid && second_valid ? LdsRaceOrdering::Ordered
                                                                : LdsRaceOrdering::Incomplete;
              }
              const auto participant = [&](uint32_t wave, uint32_t site) {
                return LdsRaceParticipant{wave,
                                          site,
                                          site + 1,
                                          (uint64_t{1} << (wave + 1)) - 1,
                                          0,
                                          4,
                                          capture_.header.manifest.sites[site].original_offset,
                                          AccessKind::Write};
              };
              const uint64_t lanes = uint64_t{first_wave + 1} * (second_wave + 1);
              LdsRaceFinding finding{LdsRaceFindingKind::CrossWave,
                                     participant(first_wave, first_site),
                                     participant(second_wave, second_site), lanes, order};
              if (order == LdsRaceOrdering::Ordered) {
                ++ordered_count;
                if (ordered.size() < examples)
                  ordered.push_back(finding);
              } else {
                ++retained_count;
                lane_pairs += lanes;
                unresolved_count += order == LdsRaceOrdering::Incomplete;
                if (retained.size() < examples)
                  retained.push_back(finding);
              }
            }
      const FailureOr<LdsRaceResult> result = analyze({examples});
      ASSERT_TRUE(result.succeeded()) << error_.message;
      const LdsRaceExecutionResult &execution = result.value().executions.front();
      EXPECT_EQ(execution.ordered_pairs, ordered_count);
      EXPECT_EQ(execution.cross_wave_pairs, retained_count);
      EXPECT_EQ(execution.cross_wave_lane_pairs, lane_pairs);
      EXPECT_EQ(execution.ordering_unresolved_pairs, unresolved_count);
      EXPECT_EQ(execution.ordered_examples, ordered);
      EXPECT_EQ(execution.cross_wave_examples, retained);
    }
  }
}

TEST_F(ConSanAnalysis, CorruptRecordDiagnosticsAreBoundedWithoutLosingIntegrityCounts) {
  constexpr uint32_t sites = 256;
  many_sites(sites);
  for (uint32_t wave = 0; wave < 2; ++wave) {
    for (uint32_t site = 0; site < sites; ++site) {
      access(wave, site, site + 1);
      wire::write_u32(capture_.executions[0].report, record(wave, site) + wire::kRecordCommit, 1);
    }
  }
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_FALSE(execution.coverage_complete);
  EXPECT_EQ(execution.rejected_records, 2 * sites);
  EXPECT_EQ(execution.integrity_events, 2 * sites + 2);
  EXPECT_EQ(execution.issues.size(), kMaxIntegrityIssueExamples);
  EXPECT_EQ(execution.cross_wave_pairs, 0u);
}

TEST_F(ConSanAnalysis, Wave64HighLanesAndFullMasksRetainExactLanePairCounts) {
  capture_.header.manifest.target = 942;
  capture_.header.manifest.wave_size = 64;
  capture_.executions.clear();
  append_execution({{128, 1, 1}, {128, 1, 1}});
  access(0, 0, 1, 0, uint64_t{3} << 62);
  access(1, 1, 1, 0, uint64_t{1} << 63);
  FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  EXPECT_TRUE(result.value().coverage_complete);
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.cross_wave_pairs, 1u);
  EXPECT_EQ(execution.cross_wave_lane_pairs, 2u);
  EXPECT_EQ(execution.collision_groups, 1u);
  EXPECT_EQ(execution.collision_lane_pairs, 1u);
  EXPECT_EQ(execution.cross_wave_examples.front().first.mask, uint64_t{3} << 62);
  EXPECT_NE(render_analysis(result.value()).find("13835058055282163712"), std::string::npos);
  access(0, 0, 1, 0, UINT64_MAX);
  access(1, 1, 1, 0, UINT64_MAX);
  result = analyze();
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.value().executions.front().collision_lane_pairs, 2016u);
  EXPECT_EQ(result.value().executions.front().cross_wave_lane_pairs, 4096u);
}

TEST_F(ConSanAnalysis, MoreThanTwoWavesShareCandidatesAndPartialWaveGapsStayLocal) {
  capture_.executions.clear();
  append_execution({{97, 1, 1}, {97, 1, 1}});
  access(0, 0, 1);
  access(1, 1, 1);
  access(2, 1, 1);
  access(3, 1, 1, 0, 2); // Only lane zero exists in the final wave.
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded()) << error_.message;
  const LdsRaceExecutionResult &execution = result.value().executions.front();
  EXPECT_EQ(execution.valid_observations, 3u);
  EXPECT_EQ(execution.rejected_records, 1u);
  EXPECT_EQ(execution.cross_wave_pairs, 2u);
  EXPECT_FALSE(execution.coverage_complete);
  EXPECT_EQ(execution.cross_wave_examples.back().second.wave, 2u);
}

TEST_F(ConSanAnalysis, Wave32CannotAttributeHighMaskBitsToExistingLanes) {
  access(0, 0, 1, 0, uint64_t{1} << 32);
  access(1, 1, 1);
  const FailureOr<LdsRaceResult> result = analyze();
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.value().executions.front().rejected_records, 1u);
  EXPECT_EQ(result.value().cross_wave_pairs, 0u);
  EXPECT_FALSE(result.value().coverage_complete);
}

} // namespace
} // namespace rocjitsu::consan
