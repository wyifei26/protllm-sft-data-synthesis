#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

BAD_STATUSES = {
    "Abnormal",
    "Cancelled",
    "Canceled",
    "Error",
    "Failed",
    "Killed",
    "Stopped",
    "StopFailed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a stage1 task until it finishes successfully, then submit the next shard."
    )
    parser.add_argument("--watch-task-id", required=True, help="Current stage1 producer task id to watch.")
    parser.add_argument(
        "--next-conf",
        type=Path,
        default=Path("/GenSIvePFS/users/yfwang/volc/task-configs/protllm/gemma4_stage1_only_8gpu_shard1.yaml"),
        help="Task YAML to submit once the watched task finishes successfully.",
    )
    parser.add_argument(
        "--success-marker",
        default="SFT_STAGE1_JOB_STARTED",
        help="Startup marker for the next shard submission verification.",
    )
    parser.add_argument(
        "--notes",
        default="protllm next shard auto-submitted after stage1 completion",
        help="Notes passed into the submit-and-watch helper.",
    )
    parser.add_argument("--poll-seconds", type=int, default=120, help="Polling interval while watching the task.")
    return parser.parse_args()


def run_cmd(argv: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(argv, capture_output=True, text=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(argv)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def fetch_task_summary(task_id: str) -> dict:
    proc = run_cmd(["volc", "ml_task", "get", "--id", task_id, "--output", "json"])
    payload = json.loads(proc.stdout.strip())
    if isinstance(payload, list):
        if not payload:
            raise RuntimeError(f"no task summary returned for {task_id}")
        return payload[0]
    if isinstance(payload, dict):
        return payload
    raise RuntimeError(f"unexpected payload for task {task_id}: {type(payload).__name__}")


def main() -> int:
    args = parse_args()
    next_conf = args.next_conf.resolve()
    if not next_conf.exists():
        raise RuntimeError(f"next shard config not found: {next_conf}")

    print(f"Watching task {args.watch_task_id} until it finishes successfully.")
    print(f"Will submit next shard with config: {next_conf}")

    while True:
        summary = fetch_task_summary(args.watch_task_id)
        status = str(summary.get("Status") or summary.get("State") or "Unknown")
        end_time = summary.get("End")
        job_name = str(summary.get("JobName") or args.watch_task_id)
        elapsed = summary.get("Elapsed")
        print(f"watch status task={job_name} status={status} elapsed={elapsed} end={end_time}")

        if status in BAD_STATUSES:
            print(f"Watched task entered terminal failure status: {status}", file=sys.stderr)
            return 1

        if end_time:
            print(f"Watched task finished successfully with status={status}. Submitting next shard.")
            submit_cmd = [
                "python",
                "/root/.codex/skills/volc-ml-task-submit/scripts/submit_and_watch_task.py",
                "--conf",
                str(next_conf),
                "--success-marker",
                args.success_marker,
                "--notes",
                args.notes,
            ]
            result = run_cmd(submit_cmd, check=False)
            if result.stdout:
                print(result.stdout.strip())
            if result.stderr:
                print(result.stderr.strip(), file=sys.stderr)
            return result.returncode

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
