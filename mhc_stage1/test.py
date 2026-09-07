#!/usr/bin/env python3
"""Run an arbitrary command with ``msprof op`` and average task duration."""

from __future__ import annotations

import argparse
import re
import shlex
import statistics
import subprocess
import sys


DURATION_RE = re.compile(r"Task Duration\(us\):\s*([0-9]+(?:\.[0-9]+)?)")
ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
FAILURE_RE = re.compile(
    r"Traceback \(most recent call last\):|Child process exited with status|"
    r"No profiling data dumped|Profiling data parse failed|"
    r"Profiling running finished\. May cause|"
    r"ERR\d+\s+UNKNOWN applica(?:it|ti)on exception",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a command repeatedly under msprof and print mean Task Duration."
    )
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--kernel-name", default="main")
    parser.add_argument("--msprof-bin", default="msprof")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help='target command, for example: "python sinkhorn_fwd.py"',
    )
    return parser.parse_args()


def normalize_command(parts: list[str]) -> list[str]:
    """Accept both a quoted command and ordinary trailing argv."""
    if parts and parts[0] == "--":
        parts = parts[1:]
    if len(parts) == 1:
        parts = shlex.split(parts[0])
    if not parts:
        raise ValueError("a target command is required")
    return parts


def task_duration(output: str) -> float | None:
    """Return the last Task Duration value printed by msprof."""
    clean_output = ANSI_ESCAPE_RE.sub("", output)
    matches = DURATION_RE.findall(clean_output)
    return float(matches[-1]) if matches else None


def main() -> int:
    args = parse_args()
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive")
    if args.timeout <= 0:
        raise ValueError("--timeout must be positive")

    target_command = normalize_command(args.command)
    command = [
        args.msprof_bin,
        "op",
        f"--kernel-name={args.kernel_name}",
        *target_command,
    ]
    print("Command:", shlex.join(command))
    if args.dry_run:
        return 0

    durations: list[float] = []
    failures = 0
    for run_index in range(1, args.repeat + 1):
        try:
            completed = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout,
                check=False,
            )
        except FileNotFoundError:
            print(f"ERROR: executable not found: {args.msprof_bin}", file=sys.stderr)
            return 2
        except subprocess.TimeoutExpired:
            failures += 1
            print(f"[{run_index}/{args.repeat}] ERROR: timeout")
            continue

        clean_output = ANSI_ESCAPE_RE.sub("", completed.stdout)
        duration = task_duration(clean_output)
        failed = (
            completed.returncode != 0
            or FAILURE_RE.search(clean_output) is not None
            or duration is None
        )
        if failed:
            failures += 1
            print(
                f"[{run_index}/{args.repeat}] ERROR: return_code="
                f"{completed.returncode}, Task Duration not usable"
            )
            print(clean_output[-2000:])
            continue

        durations.append(duration)
        print(f"[{run_index}/{args.repeat}] Task Duration: {duration:.6f} us")

    if durations:
        print(
            f"Mean Task Duration: {statistics.fmean(durations):.6f} us "
            f"({len(durations)}/{args.repeat} successful)"
        )
    else:
        print("Mean Task Duration: unavailable")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
