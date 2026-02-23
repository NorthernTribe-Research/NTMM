from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full training + distillation pipeline.")
    parser.add_argument("--config", default="mcp.json")
    parser.add_argument("--skip-prepare-data", action="store_true")
    parser.add_argument("--prepare-max-samples", type=int, default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_step(root: Path, *cmd_parts: str) -> None:
    cmd = [sys.executable, *cmd_parts]
    print("\n>>> {}".format(" ".join(cmd)))
    subprocess.run(cmd, cwd=str(root), check=True)


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    if not args.skip_prepare_data:
        prepare_cmd = ["src/prepare_data.py", "--config", args.config, "--skip-if-exists", "--seed", str(args.seed)]
        if args.prepare_max_samples is not None:
            prepare_cmd.extend(["--max-samples", str(args.prepare_max_samples)])
        run_step(root, *prepare_cmd)

    teacher_cmd = ["src/train_teacher.py", "--config", args.config, "--seed", str(args.seed)]
    if args.max_train_samples is not None:
        teacher_cmd.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_validation_samples is not None:
        teacher_cmd.extend(["--max-validation-samples", str(args.max_validation_samples)])
    run_step(root, *teacher_cmd)

    student_cmd = ["src/distil_student.py", "--config", args.config, "--seed", str(args.seed)]
    if args.max_train_samples is not None:
        student_cmd.extend(["--max-train-samples", str(args.max_train_samples)])
    if args.max_validation_samples is not None:
        student_cmd.extend(["--max-validation-samples", str(args.max_validation_samples)])
    run_step(root, *student_cmd)

    eval_cmd = ["src/evaluate_student.py", "--config", args.config]
    if args.max_test_samples is not None:
        eval_cmd.extend(["--max-test-samples", str(args.max_test_samples)])
    run_step(root, *eval_cmd)

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
