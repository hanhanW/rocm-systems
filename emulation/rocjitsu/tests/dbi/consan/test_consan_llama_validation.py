"""Fail-closed admission for exact corpus test-backend-ops cases."""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import consan_llama_validation as llama
import consan_validation as validation

CASE = "RMS_NORM(type=f32,ne=[1025,5,4,3],v=1,eps=0.000001,inplace=0)"
FIELDS = (
    "backend_name",
    "op_name",
    "op_params",
    "test_mode",
    "supported",
    "error_message",
    "backend_reg_name",
)


def output_row(**changes: str) -> str:
    row = dict(zip(FIELDS, ("ROCm0", "RMS_NORM", CASE[9:-1], "test", "1", "", "")))
    row.update(changes)
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=FIELDS, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerow(row)
    return output.getvalue()


class LlamaBackendOpsTest(unittest.TestCase):
    def test_accepts_one_exact_cpu_reference_comparison(self) -> None:
        row = llama._parse_backend_op_result(output_row(), CASE)
        self.assertEqual(row["backend_name"], "ROCm0")
        self.assertEqual(row["supported"], "1")

    def test_rejects_missing_wrong_unsupported_or_duplicate_results(self) -> None:
        for output in (
            "",
            '"backend_name","supported"\n"ROCm0","1"\n',
            output_row(backend_name="CPU"),
            output_row(test_mode="perf"),
            output_row(op_name="SOFT_MAX"),
            output_row(op_params="ne=[1]"),
            output_row(supported="0"),
            output_row(error_message="test failed"),
            output_row(error_message="not supported"),
            output_row() + output_row(),
            output_row() + '"truncated"\n',
        ):
            with self.subTest(output=output), self.assertRaises(ValueError):
                llama._parse_backend_op_result(output, CASE)

    def test_child_failure_cannot_pass_the_oracle(self) -> None:
        for returncode in (0, 1, -11, 124):
            with self.subTest(
                returncode=returncode
            ), tempfile.TemporaryDirectory() as temp:
                result_path = Path(temp) / "oracle.json"
                with mock.patch.object(
                    llama, "_run", return_value=(returncode, 1.0, output_row())
                ) as run, mock.patch.dict(
                    os.environ, {"CONSAN_ROW_RESULT_PATH": str(result_path)}
                ), contextlib.redirect_stdout(
                    io.StringIO()
                ):
                    status = llama.main(
                        [
                            "--executable",
                            "/native/test-backend-ops",
                            "--backend-op-case",
                            CASE,
                            "--output-dir",
                            temp,
                        ]
                    )
                self.assertEqual(status == 0, returncode == 0)
                payload = json.loads(result_path.read_text())
                self.assertEqual(
                    payload["oracle"], "pass" if returncode == 0 else "fail"
                )
                self.assertEqual(
                    run.call_args.args[0],
                    [
                        "/native/test-backend-ops",
                        "test",
                        "-o",
                        CASE,
                        "-b",
                        "ROCm0",
                        "--output",
                        "csv",
                    ],
                )

    def test_current_corpus_layout_routes_and_records_each_exact_case(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            build = workspace / "rocjitsu-test-corpus-build/llama-gfx1100"
            executable = build / "test-backend-ops"
            executable.parent.mkdir(parents=True)
            executable.touch()
            for relative in (
                "libggml.so",
                "libggml-base.so",
                "libggml-cpu.so",
                "ggml-hip/libggml-hip.so",
            ):
                path = build / "ggml/src" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            with mock.patch.dict(os.environ, {}, clear=True):
                for name in ("llama-rms-norm", "llama-softmax", "llama-mul-mat-q4"):
                    workload = validation._workload_for_target("gfx1100", name)
                    files = validation._input_files(workspace, "gfx1100", workload)
                    self.assertEqual(files["executable"], executable)
                    self.assertEqual(
                        files["case"],
                        workspace
                        / "rocjitsu-test-corpus/corpus/llama/selected_llama_backend_ops_tests.json",
                    )
                    self.assertEqual(
                        files["ggml-hip"], build / "ggml/src/ggml-hip/libggml-hip.so"
                    )
                    for phase in ("clean", "fault"):
                        command = validation._workload_command(
                            workspace,
                            "gfx1100",
                            workload,
                            phase,
                            workspace / "result.json",
                        )
                        self.assertEqual(
                            command[command.index("--executable") + 1], str(executable)
                        )
                        self.assertEqual(
                            command[command.index("--backend-op-case") + 1],
                            workload.llama_backend_op_case,
                        )
                (build / "ggml/src/ggml-hip/libggml-hip.so").unlink()
                with self.assertRaises(validation.ValidationError):
                    validation._llama_runtime(workspace, "gfx1100", "test-backend-ops")


if __name__ == "__main__":
    unittest.main()
