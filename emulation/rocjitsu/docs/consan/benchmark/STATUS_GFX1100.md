# ConSan `gfx1100` benchmark status

For each mode, **Startup** sums instrumentation and the measured first run;
**Run** is the absolute second-run latency followed by its ratio to the
matching uninstrumented second run. PyTorch/Gluon use synchronized host
timing. hipBLASLt uses GPU event timing, so its Startup excludes host client
setup and is not an end-to-end cold-start measurement.

| Workload | Uninstrumented Startup | Uninstrumented Run | Default Mode Startup | Default Mode Run | Default Mode (high) Startup | Default Mode (high) Run | SuperCollider Startup | SuperCollider Run | Progress |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch synthetic dense prefill (32-token prompt) | see repeated series | see repeated series | see repeated series | variable; see below | see repeated series | see repeated series | not selected | not selected | clean checks passed; 2/5 campaigns meet the native drift limit |
| PyTorch synthetic dense decode (one continuous-batch tick) | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| PyTorch synthetic four-expert top-1 MoE prefill (16-token prompt) | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |
| Gluon verified shared-memory round trip (1024 elements) | 0.442 s | 0.000107 s (1×) | 0.446 s | 0.000139 s (1.29×) | 0.448 s | 0.000134 s (1.25×) | 0.447 s | 0.00011 s (1.02×) | native validation accepted; drift reviewed below |
| hipBLASLt/Tensile verified FP16 GEMM (512×512×512) | pending | pending | pending | pending | pending | pending | pending | pending | not started: pending |

The Gluon probe used a physical Radeon PRO W7900 with native `gfx1100` code.
All three modes passed the numeric oracle and patched 32/32 applicable sites.
The kernel uses one wave32 wave; it does not qualify
cross-wave fault detection or Aorta model performance.

The table uses an independent repeat measurement. Native validation drift was
+0.75% for Run1 and +0.31% for Run2, within the prospectively recorded 5% review
limit. An earlier measurement was excluded from performance admission because
Run2 drift was -12.96%, despite the runner accepting each cell.
The approximately 0.1 ms second-run times are
small-probe measurements; the current runner records drift without enforcing
an automatic threshold, so final admission still requires this review.

Use the maintained [benchmark workflow](BENCHMARK.md) for reproduction and
retain exact runtime versions, commands, and raw measurements in execution
records. These measurements do not establish performance across other workload
shapes, toolchains, or devices.

## Dense prefill: clean checks and repeated timing

The fixed Aorta workload is a synthetic BF16 decoder with four layers, hidden
size 512, batch size one, and a 32-token prompt on the same native gfx1100 GPU.
Five independent campaigns used unchanged source, executables, inputs, and
profile controls. Each ran native discovery, two native references,
Default/default, Default/high, and final native validation in fresh processes.
The two operations in each of these 30 processes passed the CPU FP32 oracle:
all greedy tokens matched, relative L2 error was at most 0.545%, and
peak-relative logit error was at most 0.320%, within the fixed 1% bounds.

All ten instrumented cells patched 1212/1212 access and 101/101 barrier sites
across five applicable code objects. The validation clean-report checker
separately accepted all ten Run1 analysis windows with complete static/dynamic
verdicts and zero diagnostics. Run2 retained instrumentation with host analysis
disabled according to the benchmark contract. Bounded report saturation was
recorded; these observations do not establish exhaustive race coverage or
fault sensitivity. SuperCollider was not selected.

The 5% absolute native drift limit was declared before the five campaigns and
applied independently to Run1 and Run2. All runner invocations returned zero,
but campaigns 2, 3, and 4 failed this performance review. Their raw timings are
retained below for diagnosis and excluded from admitted overhead ratios.

| Campaign | Native Run2 (ms) | Default Run2 (ms) | Default/high Run2 (ms) | Native drift: Run1 / Run2 | Performance review |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 2.56 | 18.9 | 70.1 | +1.44% / -3.08% | admitted; ratios 7.39× / 27.4× |
| 2 | 2.52 | 12.6 | 69.0 | +5.75% / -6.07% | excluded: native drift |
| 3 | 2.39 | 12.3 | 70.4 | +0.995% / +59.4% | excluded: native drift |
| 4 | 3.04 | 8.18 | 73.2 | -0.649% / -6.32% | excluded: native drift |
| 5 | 2.40 | 8.71 | 69.2 | -0.140% / +2.22% | admitted; ratios 3.63× / 28.8× |

Observed Startup across all attempts was 6.72–6.94 s for Default/default and
6.73–6.84 s for Default/high. Even the two drift-admitted campaigns differ by
2.17× in Default/default Run2 latency, so the ledger does not select one as a
stable performance result or average away the variation.

Default/default retained 317–763 access records per campaign; Default/high
retained 3102–3550. Reader identities and the sampled observations varied
between fresh processes. The workgroup selector can depend on reader identity,
making sampling a concrete hypothesis to test, but these counts alone do not
prove the cause of the latency spread. Report allocation was 1,634,944 bytes
with no allocation, capacity, or cleanup failures in all ten instrumented cells.

These results reproduce substantial timing variation without a source change;
they do not establish or rule out a regression between earlier source states.
The timing follow-up should stabilize the native comparator and compare explicitly
controlled sampling at the fixed shape before expanding prompt length or batch
size. It does not block other workload validation: the immediate priority is the
[coverage expansion](../validation/STATUS_RDNA3.md#coverage-expansion-and-remaining-blockers)
and classification of missing prerequisites or features. Any source A/B
must use the same base and one declared difference. An initial environment setup
attempt failed native discovery because the matching PyTorch device wheel was
missing; after installing it and passing a GPU tensor smoke, all five declared
campaigns ran. That setup failure is retained separately from the timing series.

## Histogram host transformation cost

Five alternating before/after pairs used the same FP32/FP64 histogram input,
Default/high controls, exact kernel allowlist, toolchain, and GPU. Each fresh
process passed the numeric oracle and clean gate. The compared hooks were built
in the same checkout; the compared implementation adds the reviewed RMW and return-proof changes. The before
hook excluded two relaxed LDS RMW accesses, while the after hook covered 7/7
accesses, so the measured work is intentionally different.

The hook-reported total host instrumentation time had medians of 34.29 ms
before and 35.25 ms after, a 1.028× ratio across five samples each. Raw samples
and alternating order are retained. This bounded transformation-cost check
covers the histogram path; it does not establish GPU latency, end-to-end
startup, the cost of higher + 256 banks, or performance of the Q4 return repair.
The latter's original transform fails and has no successful matched denominator.
