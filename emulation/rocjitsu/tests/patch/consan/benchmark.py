# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compare real ConSan inline/callable observations with physical GPU timestamps."""

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import subprocess
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--rocm', type=Path, required=True)
    parser.add_argument('--ccache', type=Path, required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--wave-size', type=int, required=True)
    parser.add_argument('--threads', type=int, default=64)
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--warmups', type=int, default=100)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if (
        not 4 <= args.threads <= 1024
        or not 1 <= args.samples <= 100000
        or not 0 <= args.warmups <= 10000
    ):
        parser.error('expected 4..1024 threads, 1..100000 samples and 0..10000 warmups')
    build, rocm, ccache = (
        path.resolve() for path in (args.build, args.rocm, args.ccache)
    )
    output = args.output.resolve()
    output.mkdir()
    source = Path(__file__).resolve().parent
    cache = (build / 'CMakeCache.txt').read_text().splitlines()
    home = next(
        line.split('=', 1)[1]
        for line in cache
        if line.startswith('CMAKE_HOME_DIRECTORY:')
    )
    if Path(home).resolve() != source.parents[2]:
        parser.error('build must belong to this checkout')
    legacy = args.target in ('gfx942', 'gfx950')
    if (args.target, args.wave_size) not in (
        ('gfx942', 64),
        ('gfx950', 64),
        ('gfx1100', 32),
        ('gfx1100', 64),
        ('gfx1201', 32),
        ('gfx1201', 64),
        ('gfx1250', 32),
    ):
        parser.error('unsupported target/wave mode')
    metadata = dict(arguments=vars(args), build=str(build), rocm=str(rocm), commands=[])
    metadata['arguments'] = {key: str(value) for key, value in vars(args).items()}

    def run(label, command):
        start = time.monotonic()
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=300, check=False
        )
        (output / (label + '.log')).write_text(result.stdout + result.stderr)
        metadata['commands'].append(
            dict(
                command=[str(item) for item in command],
                exit=result.returncode,
                seconds=time.monotonic() - start,
            )
        )
        (output / 'commands.json').write_text(json.dumps(metadata, indent=2))
        result.check_returncode()
        return result.stdout

    summary = {}
    for name, case, pairs in (
        ('pair-1', 2, 1),
        ('pair-4', 2, 4),
        ('pair-16', 2, 16),
        ('four-address-groups', 10, 1),
    ):
        directory = output / name
        directory.mkdir()
        original = directory / 'original.hsaco'
        obj = directory / 'original.o'
        flags = [
            '-mwavefrontsize64' if args.wave_size == 64 else '-mno-wavefrontsize64'
        ]
        if legacy:
            flags = []
        run(
            name + '-assemble',
            [
                str(ccache),
                str(rocm / 'lib/llvm/bin/amdclang++'),
                '-x',
                'assembler-with-cpp',
                '-P',
                '-target',
                'amdgcn-amd-amdhsa',
                '-mcpu=' + args.target,
                '-mcode-object-version=5',
                *flags,
                '-DCONSAN_TARGET="amdgcn-amd-amdhsa--' + args.target + '"',
                '-DCONSAN_WAVE_SIZE=' + str(args.wave_size),
                '-DCONSAN_LEGACY=' + str(int(legacy)),
                '-DCONSAN_GFX12=' + str(int(args.target in ('gfx1201', 'gfx1250'))),
                '-DCONSAN_CASE=' + str(case),
                '-DCONSAN_BENCHMARK_PAIRS=' + str(pairs),
                '-DCONSAN_GROUP_BYTES=128',
                '-c',
                str(source / 'fixtures/native_lds.s'),
                '-o',
                str(obj),
            ],
        )
        run(
            name + '-link',
            [
                str(rocm / 'lib/llvm/bin/ld.lld'),
                '-shared',
                str(obj),
                '-o',
                str(original),
            ],
        )
        resources = {}
        for backend in ('inline', 'calls'):
            report = run(
                name + '-prepare-' + backend,
                [
                    str(build / 'tools/rj_consan_prepare'),
                    str(original),
                    args.target,
                    'lds_producer_consumer',
                    str(directory / backend),
                    '--probe-backend',
                    backend,
                ],
            )
            if 'backend=' + backend not in report.split():
                raise ValueError('preparation did not acknowledge the selected backend')
            resources[backend] = dict(
                token.split('=', 1) for token in report.splitlines()[0].split()
            )
        for variant in ('original', 'inline', 'calls'):
            image = (
                original
                if variant == 'original'
                else directory / variant / 'prepared.hsaco'
            )
            assembly = run(
                name + '-disassemble-' + variant,
                [str(rocm / 'lib/llvm/bin/llvm-objdump'), '-d', str(image)],
            )
            calls = len(re.findall(r'\bs_(?:swappc_b64|swap_pc_i64)\b', assembly))
            expected = int(resources['calls']['sites']) + 2 if variant == 'calls' else 0
            if calls != expected:
                raise ValueError(
                    f'{variant}: expected {expected} calls, decoded {calls}'
                )
            sizes = run(
                name + '-size-' + variant,
                [str(rocm / 'lib/llvm/bin/llvm-objdump'), '-h', str(image)],
            )
            text_bytes = int(
                re.search(r'^\s*\d+\s+\.text\s+([0-9a-fA-F]+)', sizes, re.MULTILINE)[1],
                16,
            )
            resources.setdefault(variant, {}).update(
                text_bytes=text_bytes, call_sites=calls
            )
            run(
                name + '-metadata-' + variant,
                [str(rocm / 'lib/llvm/bin/llvm-readobj'), '--notes', str(image)],
            )
        measured = run(
            name + '-measure',
            [
                str(build / 'tests/patch/consan/consan_dbi_benchmark'),
                str(original),
                'lds_producer_consumer',
                args.target,
                str(args.wave_size),
                str(args.threads),
                str(directory / 'inline'),
                str(directory / 'calls'),
                str(directory / 'timing'),
                str(args.samples),
                str(args.warmups),
            ],
        )
        expected_records = (
            pairs * 2 * ((args.threads + args.wave_size - 1) // args.wave_size)
            if case == 2
            else 4
        )
        for backend in ('inline', 'calls'):
            match = re.search(
                r'variant=' + backend + r' committed_records=(\d+) row_bytes=(\d+)',
                measured,
            )
            if match is None or int(match[1]) != expected_records:
                raise ValueError('benchmark did not publish the expected observations')
            resources[backend].update(
                committed_records=int(match[1]), row_bytes=int(match[2])
            )
        with (directory / 'timing/samples.csv').open() as file:
            rows = [row for row in csv.DictReader(file) if row['warmup'] == '0']
        timings = {}
        for variant in ('original', 'inline', 'calls'):
            selected = [row for row in rows if row['variant'] == variant]
            if len(selected) != args.samples:
                raise ValueError('missing measurement samples')
            timings[variant] = {}
            for metric in ('gpu_ns', 'host_launch_ns'):
                values = sorted(float(row[metric]) for row in selected)
                timings[variant][metric] = dict(
                    samples=len(values),
                    minimum=min(values),
                    median=statistics.median(values),
                    mean=statistics.mean(values),
                    p95=values[int(0.95 * (len(values) - 1))],
                    maximum=max(values),
                )
        summary[name] = dict(resources=resources, timings=timings)
        (output / 'summary.json').write_text(json.dumps(summary, indent=2))
        print(name, json.dumps(timings), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
