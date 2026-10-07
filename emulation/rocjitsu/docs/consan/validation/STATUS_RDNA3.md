# ConSan RDNA3 (`gfx1100`) status

The physical results below use native `gfx1100` code objects on a Radeon PRO
W7900. For the six hip-moi rows, baseline, Default, and SuperCollider clean
runs passed with complete applicable coverage. The 15 fault campaigns covered
17 profile or operating-point outcomes and 136 admitted and reached trials.
Each outcome used eight trials;
the table records the selected profile controls, detection counts, and
lower-preset misses needed to interpret each grade.

## P0 qualification and sensitivity

Default and SuperCollider are detector profiles, not workload priorities. The
evaluated external P0 rows are one rocBLAS SGEMM workload and three PyTorch
workloads: mode, sort, and norm/softmax. Both profiles have clean controls for
all four rows. The seven rocBLAS fault matrices and the PyTorch operation-level
campaigns below are separate evidence sets. The P0 fixture and Qwen prefill
rows in the status table are separate workloads.

Fault qualification checks whether a detector reports a deliberately introduced,
reviewed synchronization error often enough to meet the qualification contract.
A passing clean run establishes output preservation and applicable coverage;
it does not establish sensitivity to a fault. Trials must prove that the
intended mutation was installed and reached, with complete evidence and healthy
pre/post probes. The campaigns below require at least six detections in eight
trials at each declared operating point.

Fresh native discovery, allowlists, and fault inventories were generated for
rocBLAS SGEMM, PyTorch mode, PyTorch sort, and PyTorch norm/softmax.
All 24 baseline/clean results passed across these four workloads, including
Default/default, Default/high, SuperCollider standard, and SuperCollider
sleep=15 controls. Applicable static coverage is recorded below. The rocBLAS
Default/default clean run collected zero access records; its high control
collected 456.

The rocBLAS fault removes the initial LDS publication barrier before peer waves
consume the published data. The fixed 64-cubed case dispatches one workgroup
of four wave32 waves. All seven eight-trial matrices below were declared before
execution and retained every miss. Each had a passing matching clean comparator;
the four additional comparators bring the total to 28 accepted clean results.
All 56 fault trials
were admitted and reached, had complete evidence, passed the numerical oracle,
and passed pre/post health probes. These are observed counts for this workload
and fault, not a general detection-rate guarantee.

| rocBLAS operating point | Detections | Runner exit | Interpretation |
| --- | ---: | ---: | --- |
| Default/default: workgroup/cell strides 256/256 | 2/8 | 1 | Below bar; six trials collected zero records |
| Default/high: workgroup/cell strides 16/16 | 8/8 | 0 | Qualified high preset |
| Default: explicit workgroup/cell strides 1/256 | 8/8 | 0 | Qualified explicit controls; standard preset remains below bar |
| SuperCollider sleep=15, delays on reads and writes | 0/8 | 1 | Below bar |
| SuperCollider sleep=15, delays on reads only | 0/8 | 1 | Below bar |
| SuperCollider sleep=127, delays on reads and writes | 0/8 | 1 | Below bar |
| SuperCollider sleep=127, delays on reads only | 0/8 | 1 | Below bar |

`Runner exit` is the validation runner's exit code for the entire eight-trial
fault campaign, not an individual workload's exit code. The runner returns 0
when the campaign is accepted and 1 when it is rejected. In this table, the
exit-1 campaigns missed the required six detections; all their trials were
admitted and reached, with passing numerical and pre/post health checks. An
exit of 1 here does not by itself indicate a workload crash.

High changes both workgroup and cell strides from 256 to 16, and retained 456
access records in each fault trial. It detected the fault 8/8 against the
predeclared 6/8 threshold. With workgroup stride 1 and cell stride still 256,
the same fault was detected 8/8 with 75 records per trial. Thus High observes
more, but this experiment does not require denser cell sampling to recover the
measured Default misses.

**Default miss mechanism.** gfx1100 prefers a literal dispatch identity;
the hook supplies the code-object reader handle. The
[workgroup gate](../../../lib/rocjitsu/src/rocjitsu/code/patch/consan/consan_runtime_workgroup_gate.cpp)
mixes its low six bits with workgroup coordinates and tests the result against
the workgroup stride and offset. The standard preset uses stride 256 and offset
zero. This SGEMM launches only workgroup (0,0,0): its two Default/default hits
had reader identity modulo 64 equal to zero, selected that workgroup, and each
retained 75 access records with 12 conflicts. The six misses had reader identity
modulo 64 equal to 16, failed the workgroup gate, and retained zero records.
The emitted gate was inspected to confirm the bypass. The kernel and installed
fault still ran; Default did not observe LDS accesses from the excluded group.
Cell sampling operates only within a selected workgroup, so changing its stride
alone cannot recover these misses. Selecting every workgroup with workgroup
stride 1 while retaining cell stride 256 produced 75 records and detections in
all eight trials, without increasing the eight banks per range. Whole-workgroup
exclusion explains these measured misses;
additional bank capacity or LDS opcode support is not needed to recover them.
Reader-allocation-dependent selection also makes cross-run detection counts
insufficient evidence of a sensitivity change without checking what was sampled.

**SuperCollider limit and remaining uncertainty.** The active replacement
kernel descriptor and disassembly were inspected, not just the retained
original text. The selected barrier is a NOP; the consumer retains an LDS load,
`s_sleep 15`, a duplicate load, a completion wait, and value comparisons.
All four SuperCollider configurations produced no mismatch and no numerical
failure.
The physical gates passed 10/10 with no skips, including
`ConSanGfx1100Physical.SuperColliderAllSupportedSites`, whose fixture requires
a positive mismatch marker. This supports a workload sensitivity limitation;
it does not independently prove every rocBLAS replay probe correct.
SuperCollider observes value instability, so an unordered access pair need not
produce a mismatch.
Per-wave timing and observed values are not retained by its sticky marker;
the exact interleaving responsible for this workload's misses remains unknown.

**Implementation decisions.** Existing independent workgroup/cell selectors
provide a measured Default workaround. A policy guaranteeing observation of
small grids, with reproducible selection independent of reader allocation, is
worth designing and benchmarking; automatically changing the standard preset
would affect overhead and needs separate evidence. Default sampling is the
current P0 investigation; SuperCollider sensitivity work is deferred. RDNA3
currently lacks
`sleep_wave` support in both the runner and
[replay delay emitter](../../../lib/rocjitsu/src/rocjitsu/code/patch/consan/targets/consan_supercollider_target_ops.cpp).
A bounded RDNA3 wave-dependent-delay prototype is a reasonable next
SuperCollider experiment, not an established fix: retain it only with matched
clean checks, instruction/state-preservation tests, measured detection gain,
and overhead measurements. Larger uniform delays and reads-only delays did not
improve detection in these trials. The proposed policy and delay support require
separate implementation and validation.

**PyTorch operation-level evidence.** These rows invoke PyTorch operations,
not entire models. Mode's primary compute kernel uses two wave32 waves; sort's
primary radix-sort kernel uses four, alongside preparation/copy dispatches.
The combined norm/softmax row invokes two distinct operations and kernels, so
its fault results must be reported separately. Native ISA review selected
publication barriers adjacent to LDS producer/consumer instructions for each
operation. These site-specific experiments do not establish sensitivity for
every barrier or every input shape.

All four operations passed native and Default/high clean controls. Mode has
244/244 applicable access and 51/51 barrier sites, sort 234/234 and 40/40,
norm alone 14/14 and 9/9, and softmax alone 8/8 and 6/6. Norm and softmax
also have accepted component-only clean controls. A zero profiler
`LDS_Block_Size` does not establish absence of LDS instructions. The combined
row's 22/22 access and 15/15 barrier sites are the sum of its two components.
The inventory runner stops after the first relevant code object, so an
inventory of the combined row alone exposes only norm's candidate barriers;
component-only inventory and ISA review were needed for softmax. Inventory
alone does not prove mutation installation or runtime reach.

The Default/high campaigns below used eight predeclared trials per selected
site and a 6/8 acceptance bar. Every listed trial admitted its mutation and
passed pre/post GPU health checks. A detector-owned runtime diagnostic proves
reach for every reported detection. `Diagnostic` is the count out of eight
planned trials, not a population-wide detection probability.

| PyTorch operation / selected fault | Default preset | Diagnostic | Runtime reach proved | Numeric oracle | Runner exit / result |
| --- | --- | ---: | ---: | --- | --- |
| `torch.mode`, adjacent LDS publication barriers | high | 8/8 | 8/8 | pass 8/8 | 0 / qualified |
| `torch.sort`, radix-sort LDS publication barrier | high | 8/8 | 8/8 | pass 8/8 | 0 / qualified |
| `torch.softmax`, LDS publication barrier | high | 8/8 | 8/8 | fail 8/8 under fault | 0 / qualified |
| `torch.linalg.vector_norm`, LDS publication barrier | high | 2/8 | 2/8 | pass 8/8 | 1 / below bar |
| `torch.linalg.vector_norm`, same barrier | higher | 5/8 | 5/8 | pass 8/8 | 1 / below bar |
| `torch.linalg.vector_norm`, same barrier | max | 5/8 | 5/8 | pass 8/8 | 1 / below bar |

The softmax fault changed the numeric output in all eight trials; the
detector's own diagnostic, rather than the failing oracle alone, counted as
detection. Runner exit 1 for norm means that the eight-trial campaign missed
the 6/8 qualification bar; it does not mean the workload crashed. Norm's
undetected trials have no independent runtime reach witness.
The mutation was installed, but installation and a passing numeric oracle do
not prove that the selected conditional barrier executed. At max, both
workgroup and cell sampling strides are one, yet only five trials produced
diagnostics. This rules out sparse sampling as a sufficient explanation for
that operating point. Scheduling or site reachability remains a hypothesis,
not an established cause. A dynamic fault-site execution witness or a
different reviewed site would distinguish them; the present evidence does not
justify a detector implementation change. The standard Default preset was
not fault-qualified for these PyTorch operations, and SuperCollider has clean
controls only.

An initial sort specification incorrectly requested a paired-barrier selector
for a singleton barrier. Admission rejected its only attempted trial before
the faulted workload ran; that rejected attempt is excluded from the table.
The corrected exact-site specification admitted all eight trials. No valid
fault trial was excluded from the counts.

This validation covers the stated clean controls,
fault matrices, and physical gates. It does not include a full project test
suite, GCC/UBSan lane, or performance benchmark.

## P1 synthetic dense prefill clean evidence

The Aorta synthetic BF16 decoder (four layers, hidden size 512, batch one,
32-token prompt) was exercised in five independent benchmark campaigns using
native execution, Default/default, and Default/high. All 60 model operations
across 30 processes passed the independent CPU FP32 logits oracle, including
exact greedy-token agreement. The validation clean-report checker separately
accepted all ten instrumented Run1 analysis windows: access 1212/1212,
barrier 101/101, five applicable code objects, complete static/dynamic verdicts,
and zero ConSan diagnostics. Run2 remains instrumented but does not run host
analysis under the benchmark contract. Report saturation was retained in the
execution evidence; this sampled clean result does not establish fault
sensitivity. No model-level fault trials or SuperCollider runs were performed.

Only two of the five campaigns met the predeclared native timing drift limit,
and Default/default runtime varied even between those two. See the separate
[gfx1100 benchmark status](../benchmark/STATUS_GFX1100.md#dense-prefill-clean-checks-and-repeated-timing)
for all outcomes and performance limitations. Benchmark acceptance and this
independent clean-report review do not promote fault qualification.

## Coverage expansion and remaining blockers

The coverage sweep uses fresh native discovery, exact-name allowlists, and
independent numeric oracles on gfx1100. Top-k and histogram now have target
admission; the llama rows use exact backend-op cases with a fail-closed CSV
oracle and recorded runtime libraries. The selected workload families are
comparable to other targets, although their priorities and specializations
vary. Aorta synthetic prefill remains separate from IREE Qwen.

| Workload | Fresh Default evidence | Remaining limitation / action |
| --- | --- | --- |
| PyTorch top-k | Native and default/high clean pass; access 457/457, barrier 97/97. Selected BF16 radix-count publication fault: high 8/8 detected and reached. | No fault trials with the `default` preset; other fault sites have not been tested. The FP64 specialization is single-wave and does not establish cross-wave spill coverage. |
| PyTorch histogram | Native and high clean pass after relaxed LDS RMW support; access 7/7, barrier 4/4. Matching higher + 256-bank clean passes; FP32 fault 6/8 and FP64 fault 8/8. | Qualification applies to those explicit controls. Lower-preset fault sensitivity and overhead are not established. |
| llama RMS norm | Native and default/high/max clean pass; access 2/2, barrier 1/1. Publication fault: high 2/8, max 6/8, all admitted and reached. | Six high misses retained zero access records. Max meets the 6/8 bar; high remains below it. No fault trials with the `higher` or `default` presets. |
| llama softmax | Native passes. High reproduces eight conflicts with access 8/8 and barrier 2/2; sparse default clean passes. | Pinned corpus source predates the [upstream scratch-reuse fix](https://github.com/ggml-org/llama.cpp/pull/26385). Backport and affected-path validation are deferred to the next round, before fault qualification. |
| llama Q4 matmul | Native passes. The original hook rejects an unresolved helper return; the return-proof repair admits the object and instruments access 102/102, barrier 6/6. High then reports 33 write/write conflicts. | Clean remains blocked; fallback tile investigation and a source control are deferred to the next round, before faults. An earlier run without diagnostics does not resolve the conflicts reported by `high`. |
| IREE Qwen prefill | Matched MLIR/parameters/reference data are provisioned. Native and high clean pass; access 6/6, barrier 6/6. Workgroup/cell strides 1/256 also pass. | Default/default produced no visible records after 57 instrumented dispatches; workload exit 86, runner exit 1. Model fault sites still need review and matching campaigns. |
| Sharktank TP1 prefill | Native and default/high clean pass; access 372/372, barrier 21/21. | No fault-injection trials in this model run; review a kernel synchronization fault and run matching clean/fault trials. |
| Sharktank TP1 decode/combined | Native and default/high clean pass; access 1120/1120, barrier 74/74. | No fault-injection trials in this model run; review a kernel synchronization fault and run matching clean/fault trials. |

All five newly executed fault campaigns used eight predeclared trials and a
six-detection acceptance bar. All 40 mutations were admitted, all trials had
complete evidence, and pre/post health checks passed. Top-k has detector-owned
reach evidence in all eight trials; its numeric oracle failed in six faulted
trials. Histogram and RMS numeric oracles passed in every fault trial. RMS/high
and histogram/FP32 misses use the reviewed unconditional ISA path as their reach
witness. Numeric failure alone never counts as a detector hit.

RMS/max retained 16 records in every trial, including its two misses, with
31–48 saturated windows. Dense sampling therefore does not guarantee detection;
retention and interleaving remain separate follow-ups. This is not evidence for
a missing LDS opcode or an instruction to keep tuning until every trial hits.

**Implemented gaps.** Histogram originally passed a 5/5 supported-access gate
while two LDS RMWs were excluded. Enabling the target's relaxed RMW access
contract and qualifying the FP64 CAS initial load raises coverage to 7/7 without
inventing synchronization. Positive and near-miss tests cover this proof.
The Q4 return proof now ignores unreachable decoded padding while still
rejecting foreign live predecessors, clobbered saved values, and invalid lane
restores. These are implementation changes, in addition to runner and status
updates; successful transformation is not a clean verdict.

**Input diagnosis.** Softmax reuses reduction scratch before all waves have
consumed the preceding MAX partials: a later SUM store can overwrite an LDS
location while another wave still reads its MAX value. The eight high-preset
reports are read/write conflict pairs from this reuse, not eight distinct
source bugs. An earlier reduced native fixture, with no DBI hook loaded,
failed its numerical oracle in 20/20 launches when one wave was delayed before
reading the partials. Adding a consumption-completion barrier passed 20/20
under the same delay. Both versions passed 20/20 without the delay. A matching
full-workload source control with the added barrier passed five instrumented
clean runs. These controls support a workload synchronization bug independently
of DBI; they do not qualify the complete corpus backport. The fresh high run
reproduces the original conflict reports; the full source repair remains
deferred to the next round.
The corpus pin predates the merged
[upstream scratch-reuse repair](https://github.com/ggml-org/llama.cpp/pull/26385).
Its vendored `common.cuh`, `softmax.cu`, and `norm.cu` still match that older
upstream source. The full patch applies without conflicts, but has not been
backported or runtime-qualified here. It includes the ordinary softmax barrier,
separate scratch for cooperative softmax and group norm, and the helper's
scratch-lifetime contract; the one-barrier control above validates only the
ordinary softmax mechanism.
Q4's new diagnostics map to two-address LDS stores in the fallback tile loader.
The source clamps multiple row indices to the same final row and uses the
clamped index for both source and LDS destination. This is a concrete aliasing
mechanism and a candidate explanation for the observed write/write pairs.
An independent source control is still needed to establish the complete cause;
no same-value suppression or new detector exception is justified by this run.

**Sampling and model scope.** Qwen's accepted workgroup-stride-1 control leaves
cell stride at 256. It supplies the missing visible evidence without new opcode
support, supporting a workgroup-selection gap. This result and RMS's empty
high misses motivate reproducible small-grid sampling controls, not an automatic
preset change. Qwen is the fixed five-token FP32 regression, and TP1 uses the
test suite's toy model; neither measures a serving system or production context
lengths. Published Qwen inputs and regeneration are described in
[Preparing Qwen](VALIDATION.md#preparing-qwen).

Qwen and TP1 exercise DBI inside a model runtime: the generated kernels, their
dispatch sequence, and the instrumentation/report lifecycle must work together.
A standalone operation test does not exercise that same integration. The model
fault test still targets a reviewed synchronization error in a selected kernel;
it does not attempt to validate every model operation or model quality.
[CDNA4](STATUS_CDNA4.md), [CDNA5](STATUS_CDNA5.md), and
[RDNA4](STATUS_RDNA4.md) already record Qwen and TP1 clean/fault evidence, with
individual failures retained. CDNA5's evidence is emulated, not physical.
Matching these registered workloads is the model coverage objective here.
Aorta synthetic prefill is additional clean/benchmark evidence; fault-qualifying
that benchmark is not required to complete the Qwen/TP1 comparison.

**Native families.** Fresh builds of all six hip-moi rows pass native and both
default/high clean controls. Applicable access/barrier counts match the table:
D128 block 278/138, D128 pressure 503/36, WMMA attention 115/18, Stream-K arrival
14/1, tree atomic-OR 18/1, and Jakub 320/13. Arrival and tree also instrument
1/1 and 2/2 atomic sites. These are 24 accepted baseline/instrumented row results,
with 13 native GTest cases per pass and no skips. The table's native-family fault
counts are retained prior qualification, not fresh trials of these rebuilt
objects. New inventories have different object identities; re-review those
identities before rerunning selected release/publication faults. This sweep
finds no missing applicable feature in these clean paths. RDNA3's documented
release-proof scope remains narrower than other targets; target-specific
cluster/tensor-DMA forms are not gfx1100 port requirements.

Verification includes 264 validation-runner and 30 benchmark-runner Python
tests, 1,206 selected ConSan/DBT host tests, the generated
capability-document check, and all ten physical compact gates without skips.
The build completed; warnings came from the unchanged GoogleTest header and
Triton assembly commands' unused ROCm-path argument. Changed C++ sources compiled
with warnings as errors. This is not a full project suite or GCC/UBSan result.

Input-correctness investigation and repair for softmax and Q4 are deferred to
the next round. Their failed evidence and red clean status remain recorded;
neither row proceeds to fault qualification until clean is established. These
row-specific blockers do not gate the current round's independent workloads.
Next prioritize reviewed Qwen/TP1 faults and explicit sampling comparisons.
Qualification needs one passing configuration per selected workload. Lower-preset
comparisons establish a minimum passing preset or inform sampling policy; an
untested `default` preset does not block a qualified `high` configuration.
Broader prefill timing work remains deferred;
validation elapsed time is not kernel latency. SuperCollider qualification is
deferred. Worktree isolation alone does not require a separate PR.

## Reproduction and status

Reproduce these rows with the maintained
[validation runner](../../../tests/dbi/consan/consan_validation.py), the
[reviewed `gfx1100` fault specification](../../../tests/dbi/consan/consan_validation_faults_gfx1100.json),
and the linked allowlist procedure below. The runner's `manifest` and `explain`
commands provide the current target-native workload commands, profile
environment, correctness oracles, and fault policy.

Keep exact commands, exit statuses, toolchain versions, binary identities,
mutation offsets, and raw reports in the execution records. Regenerate native
inventories and review fault sites for the binaries under test. This status
document describes workload behavior and qualification limits independently
of a particular checkout, result directory, or revision.

The validation registry resolves all six hip-moi rows to target-native GFX11
fixtures and fails closed instead of substituting another architecture. A
profile is green only when its clean and reviewed fault campaigns meet the
shared qualification contract.

Start any new revalidation with
[rocprofv3-based allowlist discovery and application](VALIDATION.md#first-step-for-revalidation-generate-and-apply-kernel-allowlists).
Apply this before retrying recorded timeouts or raising deadlines; update their
status only after new runs provide evidence.

Physical evidence uses native `gfx1100` code objects on a matching GPU;
simulator prerequisites use RocJITsu `configs/gfx1100_w7900.json`. Exact counts
belong to the binaries used for qualification and must be refreshed after
relevant source, toolchain, workload, emulator, or runtime changes.

The [native publication tests](../../../tests/dbi/consan/hip_consan_rdna3_test.hip)
check output preservation with and without instrumentation. The aligned
32-bit store case must produce a complete publication trace. Unaligned stores
execute their original instruction because observation through atomic exchange
requires natural alignment; they must set the sticky dropped marker so the
decoder rejects the trace as incomplete. The tests keep the event count below
capacity so overflow detection cannot mask a missing dropped marker.

## How to read the qualification table

The table answers two questions for each workload and detector configuration:
does the unmodified workload pass the clean checks, and does ConSan detect the
reviewed synchronization faults introduced into that workload? It records
validation evidence, not benchmark performance or a guarantee of detecting all
possible bugs. P0/P1 are workload priorities, not pass/fail grades.

- **Clean** means no fault was deliberately injected. A clean pass requires
  correct output, the required instrumentation/report checks, and no unexpected
  diagnostics. It does not prove that the workload is race-free: sampling may
  miss a conflict. If another preset reports conflicts on the same unmodified
  workload, those reports must be resolved before declaring the row clean.
- **Fault** means a reviewed synchronization error was deliberately introduced
  into a kernel. `8/8` means ConSan detected that selected fault in all eight
  trials. The acceptance bar is at least `6/8`, with matching clean checks,
  proof that the mutation was installed and executed, and healthy execution.
  For a model row, the faulted kernel must run inside that model workload;
  passing a standalone operation's fault test does not establish this result.
- **No fault trials** means the experiment has not been run for the stated
  workload or preset. It is different from `0/8`, where eight trials were run
  and none detected the fault. It does not by itself identify a missing feature.
- **Access/barrier counts** describe instrumented versus applicable static
  instruction sites. `6/6` sites does not mean six fault detections or that
  every runtime workgroup/access was sampled.

The **Default detector** column names the engine. Within that column,
`default`, `high`, `higher`, and `max` name its sampling presets. A green
`high` result qualifies that configuration; it does not qualify the `default`
preset. The text names untested presets and measured misses separately.
SuperCollider is a separate detector whose further qualification is deferred.

The main table reports the selected passing configuration, or the concrete
blocker when none qualifies. Following [CDNA4](STATUS_CDNA4.md), **lowest
passing** is used only when every lower preset from `default` was tested and
failed the qualification contract. Other green rows name a **qualified
configuration**, without claiming it is minimal. No additional `default`-preset
fault campaign is required solely to keep a qualified row green. Lower-preset
results and untested settings belong in the investigation notes above, not as
separate blockers in the main table.

For the retained native-family calibration, `default` detected D128 block 2/8,
D128 pressure 4/8, and WMMA attention 4/8; each reached 8/8 with `high`.
Jakub detected 0/8 with `default`, 1/8 with `high`, and 8/8 with `higher`.
These are the historical fault results described under native families, not
new trials of the rebuilt objects. rocBLAS and RMS lower-preset results remain
in their investigation sections above.

Shared [color scale](VALIDATION.md#status-colors): 🟩 qualified; 🟨 clean run
established, but fault qualification is pending/below bar or the workload is
outside detector scope; 🟧 clean qualification blocked by prerequisites,
unsupported applicable operations, incomplete coverage/evidence, or a timeout;
🟥 observed correctness or instrumentation failure. Empty/🩶 means unassessed
for this execution target; simulator prerequisites alone do not qualify hardware.

| Set | Priority | Workload / validation ID | Default detector | SuperCollider detector |
| --- | ---: | --- | --- | --- |
| Physical compact gate | P0 | native two-wave LDS fixture (`ConSanGfx1100Physical.*`) | 🟨 exact clean output, visible records, zero diagnostics; reviewed conflict and broad campaign missing | 🟨 exact clean/all-sites rows and mutation containment; broad E2E fault campaign missing |
| Simulator prerequisite | P0 | native two-wave LDS fixture (`ConSanGfx1100Sim.*`) | 🩶 prerequisite passes | 🩶 prerequisite passes |
| Broad E2E | P0 | Qwen3-0.6B prefill (`qwen-prefill`) | 🟨 `high`: clean pass; access 6/6, barrier 6/6. No fault trials in this model run; kernel fault-site review and trials remain. | 🩶 deferred |
| Broad E2E | P0 | rocBLAS SGEMM 64³ (`rocblas-sgemm-square-64`) | 🟩 high (lowest passing): clean pass; access 232/232, barrier 5/5; initial-publication fault 8/8 | 🟨 clean pass; access 232/232; sleep=15 and sleep=127, each with all-access or reads-only delay: four separate 0/8 batches (bar 6/8) |
| Broad E2E | P0 | PyTorch mode (`pytorch-torch-mode`) | 🟩 `high`: clean pass; access 244/244, barrier 51/51; selected publication fault detected and reached 8/8 | 🟨 clean pass; access 244/244; no fault trials |
| Broad E2E | P0 | PyTorch sort (`pytorch-torch-sort`) | 🟩 `high`: clean pass; access 234/234, barrier 40/40; selected publication fault detected and reached 8/8 | 🟨 clean pass; access 234/234; no fault trials |
| Broad E2E | P0 | PyTorch norm/softmax (`pytorch-norm-softmax`) | 🟨 combined clean pass; access 22/22, barrier 15/15. Component fault trials: softmax high 8/8 detected/reached; norm high 2/8, higher 5/8, max 5/8 diagnostics with miss reach unproved; combined row below bar | 🟨 combined clean pass; access 22/22; no fault trials |
| Model clean check | P1 | Aorta synthetic dense prefill (benchmark ID `pytorch-dense-prefill`) | 🟨 `default`/`high`: 10/10 Run1 clean-report checks; access 1212/1212, barrier 101/101; both operations pass CPU oracle. Additional benchmark evidence; no fault qualification claimed. | 🩶 not selected |
| Broad E2E | P1 | PyTorch top-k (`pytorch-torch-topk`) | 🟩 `high`: clean pass; access 457/457, barrier 97/97; BF16 publication fault 8/8, all reached | 🩶 deferred |
| Broad E2E | P1 | PyTorch histogram (`pytorch-torch-histc`) | 🟩 higher + 256 banks: clean pass; access 7/7, barrier 4/4; FP32 fault 6/8 and FP64 8/8, all admitted/reached | 🩶 deferred |
| Broad E2E | P1 | llama.cpp RMS norm (`llama-rms-norm`) | 🟩 max: clean pass; access 2/2, barrier 1/1; fault 6/8, all reached | 🩶 deferred |
| Broad E2E | P1 | llama.cpp softmax (`llama-softmax`) | 🟥 Known workload scratch-reuse race, [fixed upstream](https://github.com/ggml-org/llama.cpp/pull/26385) but absent from the corpus pin. `high`: eight read/write conflicts without fault injection; access 8/8, barrier 2/2. Backport and validation deferred to next round. | 🩶 deferred |
| Broad E2E | P1 | llama.cpp Q4 matmul (`llama-mul-mat-q4`) | 🟥 return-proof repair admits transformation; high clean reports 33 write/write conflicts, access 102/102, barrier 6/6; input investigation deferred to next round | 🩶 deferred |
| Broad E2E | P1 | Sharktank TP1 prefill (`tp1-prefill`) | 🟨 Clean: native and `default`/`high` pass; access 372/372, barrier 21/21. Fault: no trials in this model run; kernel fault-site review and trials remain. | 🩶 deferred |
| Broad E2E | P1 | Sharktank TP1 decode/combined (`tp1-decode-combined`) | 🟨 Clean: native and `default`/`high` pass; access 1120/1120, barrier 74/74. Fault: no trials in this model run; kernel fault-site review and trials remain. | 🩶 deferred |
| Broad E2E | P2 | Sharktank TP2 family (`tp2-family`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P3 | Sharktank CLIP BF16 (`clip-bf16`) | 🩶 unassessed | 🩶 unassessed |
| Broad E2E | P4 | hip-moi D128 block (`d128-block`) | 🟩 high (lowest passing): clean pass; access 278/278, barrier 138/138; grouped K-publication fault 8/8 | 🟩 sleep=15: clean pass; access 278/278; grouped K-publication fault 8/8 |
| Broad E2E | P4 | hip-moi D128 pressure (`d128-pressure`) | 🟩 high (lowest passing): clean pass; access 503/503, barrier 36/36; grouped K/V-publication fault 8/8 | 🟨 sleep=15: clean pass; access 503/503; grouped K/V-publication fault 0/8 (bar 6/8) |
| Broad E2E | P4 | hip-moi WMMA attention (`wmma-attention`) | 🟩 high (lowest passing): clean pass; access 115/115, barrier 18/18; grouped K/V-publication fault 8/8 | 🟩 sleep=15: clean pass; access 115/115; grouped K/V-publication fault 8/8 |
| Broad E2E | P4 | hip-moi Stream-K arrival (`streamk-arrival`) | 🟩 high: clean pass; access 14/14, barrier 1/1, atomic 1/1; release-order fault 8/8 | 🟨 delay=15: clean pass; access 14/14; release-order fault 0/8 (bar 6/8) |
| Broad E2E | P4 | hip-moi tree atomic-OR (`tree-atomic-or`) | 🟩 high: clean pass; access 18/18, barrier 1/1, atomic 2/2; producer release-order fault 8/8 | 🟨 delay=15: clean pass; access 18/18; producer release-order fault 0/8 (bar 6/8) |
| Broad E2E | P4 | hip-moi Jakub attention (`jakub-attention`) | 🟩 higher (lowest passing): clean pass; access 320/320, barrier 13/13; load-to-compute publication fault 8/8 | 🟨 sleep=15: clean pass; access 320/320; load-to-compute publication fault 0/8 (bar 6/8) |
