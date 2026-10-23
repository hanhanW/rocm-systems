// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "hsa_test_util.h"
#include "rocjitsu/code/consan/encoding.h"
#include "rocjitsu/code/consan/image/artifact.h"
#include "rocjitsu/code/patch/kernarg_extension.h"

#include <array>
#include <charconv>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <span>
#include <string>
#include <vector>

namespace {

using namespace rocjitsu;
using namespace rocjitsu::consan;

bool require(hsa_status_t status, const char *operation) {
  if (status == HSA_STATUS_SUCCESS)
    return true;
  std::cerr << operation << " failed: HSA status " << uint32_t{status} << '\n';
  return false;
}

// This harness loads the prepared image directly. Profiling is confined to
// its own queue and never relaxes the capture plugin's profiling admission.
class Benchmark {
public:
  ~Benchmark() { close(); }

  bool initialize(std::string_view target) {
    if (!require(hsa_init(), "HSA initialization"))
      return false;
    initialized_ = true;
    agents_ = test::agents(target);
    if (!agents_.cpu.handle || !agents_.gpu.handle)
      return false;
    pool_ = test::fine_pool(agents_.cpu);
    return pool_.handle &&
           require(hsa_queue_create(agents_.gpu, 128, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, 0, 0,
                                    &queue_),
                   "queue creation") &&
           require(hsa_amd_profiling_set_profiler_enabled(queue_, 1), "queue profiling") &&
           require(hsa_signal_create(1, 0, nullptr, &signal_), "completion signal") &&
           require(hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &frequency_),
                   "timestamp frequency");
  }

  bool load(size_t index, std::span<const uint8_t> image, const std::string &entry,
            const Manifest *manifest, uint32_t threads) {
    Variant &variant = variants_.at(index);
    variant.manifest = manifest;
    variant.threads = threads;
    if (!require(
            hsa_code_object_reader_create_from_memory(image.data(), image.size(), &variant.reader),
            "code reader") ||
        !require(hsa_executable_create_alt(HSA_PROFILE_FULL,
                                           HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr,
                                           &variant.executable),
                 "executable creation") ||
        !require(hsa_executable_load_agent_code_object(variant.executable, agents_.gpu,
                                                       variant.reader, nullptr, nullptr),
                 "load") ||
        !require(hsa_executable_freeze(variant.executable, nullptr), "freeze"))
      return false;
    hsa_executable_symbol_t symbol{};
    uint32_t kernarg_bytes = 0;
    if (!require(hsa_executable_get_symbol_by_name(variant.executable, (entry + ".kd").c_str(),
                                                   &agents_.gpu, &symbol),
                 "kernel symbol") ||
        !require(hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                                &variant.kernel),
                 "kernel object") ||
        !require(
            hsa_executable_symbol_get_info(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &variant.group_bytes),
            "group size") ||
        !require(hsa_executable_symbol_get_info(
                     symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE,
                     &variant.private_bytes),
                 "private size") ||
        !require(
            hsa_executable_symbol_get_info(
                symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kernarg_bytes),
            "kernarg size") ||
        variant.private_bytes != 0 || (!manifest && kernarg_bytes != 16) ||
        (manifest && manifest->entry.original_kernarg_bytes != 16))
      return false;
    if (!allocate(16, variant.arguments) || !allocate(4096, variant.output))
      return false;
    if (manifest) {
      const LaunchDimensions dimensions{{threads, 1, 1}, {threads, 1, 1}};
      const FailureOr<LaunchLayout> layout =
          plan_launch(*manifest, dimensions, variant.group_bytes, 0);
      if (layout.failed() || layout.value().wrapper_bytes != kernarg_bytes)
        return false;
      variant.layout = layout.value();
      if (!allocate(variant.layout.wrapper_bytes, variant.wrapper) ||
          !allocate(kContextBytes, variant.context) ||
          !allocate(variant.layout.report_bytes, variant.report))
        return false;
    }
    return true;
  }

  bool measure(size_t index, uint64_t serial, double &gpu_ns, uint64_t &host_ns,
               std::vector<uint8_t> &rows, std::vector<uint8_t> &output) {
    Variant &variant = variants_.at(index);
    std::memset(variant.arguments, 0, 16);
    std::memset(variant.output, 0, 4096);
    constexpr uint32_t kInput = 7;
    std::memcpy(variant.arguments, &kInput, sizeof(kInput));
    std::memcpy(static_cast<uint8_t *>(variant.arguments) + 8, &variant.output, sizeof(void *));
    if (variant.manifest) {
      const LaunchDimensions dimensions{{variant.threads, 1, 1}, {variant.threads, 1, 1}};
      const auto transport =
          initialize_transport(*variant.manifest, dimensions, variant.group_bytes, 0, serial,
                               reinterpret_cast<uint64_t>(variant.report));
      constexpr std::array payload_layout{KernargExtensionPayloadLayout{8, 8}};
      const std::optional<KernargExtensionLayout> wrapper =
          make_kernarg_extension_layout(16, payload_layout);
      const uint64_t context = reinterpret_cast<uint64_t>(variant.context);
      const std::array payloads{KernargExtensionPayloadWrite{&context, sizeof(context)}};
      if (transport.failed() || !wrapper ||
          !write_kernarg_extension_wrapper(
              {static_cast<uint8_t *>(variant.wrapper), variant.layout.wrapper_bytes}, *wrapper,
              variant.arguments, reinterpret_cast<uint64_t>(variant.arguments), payloads))
        return false;
      std::memcpy(variant.context, transport.value().context.data(), kContextBytes);
      std::memcpy(variant.report, transport.value().report.data(), variant.layout.report_bytes);
    }
    hsa_signal_store_screlease(signal_, 1);
    const std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    const uint64_t write = hsa_queue_add_write_index_relaxed(queue_, 1);
    auto *packet = static_cast<hsa_kernel_dispatch_packet_t *>(queue_->base_address) +
                   (write & (queue_->size - 1));
    std::memset(packet, 0, sizeof(*packet));
    packet->setup = 1;
    packet->workgroup_size_x = static_cast<uint16_t>(variant.threads);
    packet->workgroup_size_y = packet->workgroup_size_z = 1;
    packet->grid_size_x = variant.threads;
    packet->grid_size_y = packet->grid_size_z = 1;
    packet->group_segment_size = variant.group_bytes;
    packet->kernel_object = variant.kernel;
    packet->kernarg_address = variant.manifest ? variant.wrapper : variant.arguments;
    packet->completion_signal = signal_;
    const uint16_t header = (HSA_PACKET_TYPE_KERNEL_DISPATCH << HSA_PACKET_HEADER_TYPE) |
                            (1u << HSA_PACKET_HEADER_BARRIER) |
                            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCACQUIRE_FENCE_SCOPE) |
                            (HSA_FENCE_SCOPE_SYSTEM << HSA_PACKET_HEADER_SCRELEASE_FENCE_SCOPE);
    __atomic_store_n(&packet->header, header, __ATOMIC_RELEASE);
    hsa_signal_store_screlease(queue_->doorbell_signal, write);
    hsa_signal_value_t completed = 1;
    const std::chrono::steady_clock::time_point deadline = start + std::chrono::seconds(20);
    while (completed == 1 && std::chrono::steady_clock::now() < deadline)
      completed = hsa_signal_wait_scacquire(signal_, HSA_SIGNAL_CONDITION_LT, 1, 1000000,
                                            HSA_WAIT_STATE_BLOCKED);
    if (completed == 1) {
      std::cerr << "benchmark timed out; retaining live GPU allocations\n";
      std::_Exit(24);
    }
    host_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - start)
                  .count();
    hsa_amd_profiling_dispatch_time_t timing{};
    if (completed != 0 ||
        !require(hsa_amd_profiling_get_dispatch_time(agents_.gpu, signal_, &timing),
                 "dispatch timestamps") ||
        timing.end <= timing.start || frequency_ == 0)
      return false;
    gpu_ns = static_cast<double>(timing.end - timing.start) * 1.e9 / frequency_;
    output.assign(static_cast<uint8_t *>(variant.output),
                  static_cast<uint8_t *>(variant.output) + 4096);
    if (variant.manifest) {
      const auto *report = static_cast<const uint8_t *>(variant.report);
      rows.assign(report + kReportHeaderBytes, report + variant.layout.report_bytes);
      for (uint32_t wave = 0; wave < variant.layout.wave_count; ++wave) {
        const size_t row = wave * variant.layout.row_stride;
        if (wire::read_u32(rows, row + wire::kRowStatus) !=
                (wire::kRowEntered | wire::kRowExited) ||
            wire::read_u64(rows, row + wire::kRowSerial) != serial)
          return false;
      }
    }
    return true;
  }

  bool close() {
    bool success = true;
    const auto check = [&](hsa_status_t status) {
      success &= require(status, "benchmark cleanup");
    };
    for (Variant &variant : variants_) {
      for (void *pointer :
           {variant.report, variant.context, variant.wrapper, variant.output, variant.arguments})
        if (pointer)
          check(hsa_amd_memory_pool_free(pointer));
      if (variant.executable.handle)
        check(hsa_executable_destroy(variant.executable));
      if (variant.reader.handle)
        check(hsa_code_object_reader_destroy(variant.reader));
      variant = {};
    }
    if (signal_.handle)
      check(hsa_signal_destroy(signal_));
    signal_ = {};
    if (queue_)
      check(hsa_queue_destroy(queue_));
    queue_ = nullptr;
    if (initialized_)
      check(hsa_shut_down());
    initialized_ = false;
    return success;
  }

private:
  struct Variant {
    hsa_executable_t executable{};
    hsa_code_object_reader_t reader{};
    uint64_t kernel = 0;
    uint32_t group_bytes = 0, private_bytes = 0, threads = 0;
    void *arguments = nullptr, *wrapper = nullptr, *context = nullptr;
    void *report = nullptr, *output = nullptr;
    const Manifest *manifest = nullptr;
    LaunchLayout layout;
  };

  bool allocate(size_t bytes, void *&pointer) {
    return require(hsa_amd_memory_pool_allocate(pool_, bytes, 0, &pointer), "allocation") &&
           require(hsa_amd_agents_allow_access(1, &agents_.gpu, nullptr, pointer), "GPU access");
  }

  bool initialized_ = false;
  test::Agents agents_{};
  hsa_amd_memory_pool_t pool_{};
  hsa_queue_t *queue_ = nullptr;
  hsa_signal_t signal_{};
  uint64_t frequency_ = 0;
  std::array<Variant, 3> variants_;
};

bool number(const char *text, uint32_t &value) {
  const std::from_chars_result parsed = std::from_chars(text, text + std::strlen(text), value);
  return parsed.ec == std::errc{} && *parsed.ptr == '\0';
}

int run(int argc, char **argv) {
  if (argc != 11) {
    std::cerr << "Usage: consan_dbi_benchmark ORIGINAL ENTRY TARGET WAVE THREADS "
                 "INLINE-ARTIFACT CALLS-ARTIFACT NEW-OUTPUT-DIRECTORY SAMPLES WARMUPS\n";
    return 2;
  }
  uint32_t wave = 0, threads = 0, samples = 0, warmups = 0;
  if (!number(argv[4], wave) || (wave != 32 && wave != 64) || !number(argv[5], threads) ||
      threads == 0 || threads > 1024 || !number(argv[9], samples) || samples == 0 ||
      samples > 100000 || !number(argv[10], warmups) || warmups > 10000)
    return 2;
  std::ifstream input(argv[1], std::ios::binary);
  const std::vector<uint8_t> original{std::istreambuf_iterator<char>(input),
                                      std::istreambuf_iterator<char>()};
  ContractError error;
  FailureOr<LoadedArtifact> inlined = load_artifact(argv[6], original, argv[3], &error);
  FailureOr<LoadedArtifact> calls = load_artifact(argv[7], original, argv[3], &error);
  if (inlined.failed() || calls.failed()) {
    std::cerr << error.message << '\n';
    return 3;
  }
  if (inlined.value().manifest.wave_size != wave || calls.value().manifest.wave_size != wave ||
      inlined.value().manifest.entry.name != argv[2] ||
      calls.value().manifest.entry.name != argv[2])
    return 3;
  const std::filesystem::path destination(argv[8]);
  if (!std::filesystem::create_directory(destination))
    return 3;
  std::ofstream measurements(destination / "samples.csv");
  measurements << "iteration,warmup,order,variant,gpu_ns,host_launch_ns\n";
  Benchmark benchmark;
  if (!benchmark.initialize(argv[3]))
    return 77;
  if (!benchmark.load(0, original, argv[2], nullptr, threads) ||
      !benchmark.load(1, inlined.value().prepared, argv[2], &inlined.value().manifest, threads) ||
      !benchmark.load(2, calls.value().prepared, argv[2], &calls.value().manifest, threads))
    return 4;
  constexpr std::array names{"original", "inline", "calls"};
  for (uint32_t iteration = 0; iteration < warmups + samples; ++iteration) {
    std::array<std::vector<uint8_t>, 3> rows, outputs;
    for (size_t order = 0; order < names.size(); ++order) {
      const size_t index = (iteration + order) % names.size();
      double gpu_ns = 0;
      uint64_t host_ns = 0;
      if (!benchmark.measure(index, iteration + 1, gpu_ns, host_ns, rows[index], outputs[index]))
        return 5;
      measurements << iteration << ',' << (iteration < warmups) << ',' << order << ','
                   << names[index] << ',' << gpu_ns << ',' << host_ns << '\n';
    }
    if (outputs[0] != outputs[1] || outputs[0] != outputs[2] || rows[1] != rows[2]) {
      std::cerr << "backend outputs or observation rows differ at iteration " << iteration << '\n';
      return 6;
    }
    if (iteration + 1 == warmups + samples) {
      for (size_t index = 1; index < names.size(); ++index) {
        const Manifest &manifest = index == 1 ? inlined.value().manifest : calls.value().manifest;
        const size_t stride =
            kRowHeaderBytes + manifest.sites.size() * kAddressGroupsPerSite * kAccessRecordBytes;
        uint64_t records = 0;
        for (size_t row = 0; row < rows[index].size(); row += stride)
          for (size_t slot = 0; slot < manifest.sites.size() * kAddressGroupsPerSite; ++slot)
            records +=
                wire::read_u32(rows[index], row + kRowHeaderBytes + slot * kAccessRecordBytes +
                                                wire::kRecordCommit) == wire::kAccessCommitted;
        std::ofstream last(destination / (std::string(names[index]) + "-last-rows.bin"),
                           std::ios::binary);
        last.write(reinterpret_cast<const char *>(rows[index].data()), rows[index].size());
        last.close();
        if (last.fail())
          return 7;
        std::cout << "variant=" << names[index] << " committed_records=" << records
                  << " row_bytes=" << rows[index].size() << '\n';
      }
    }
  }
  measurements.close();
  if (measurements.fail() || !benchmark.close())
    return 7;
  std::cout << "samples=" << samples << " warmups=" << warmups
            << " variants=3 outputs_and_observations_equal=true\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) { return run(argc, argv); }
