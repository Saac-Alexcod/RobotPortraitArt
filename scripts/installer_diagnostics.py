"""Bounded, line-logged subprocess runner for the Windows first-run installer.

Sends each child line to the console and a step-specific UTF-8 log. It emits a
heartbeat while a process is silent and kills its process tree on timeout.
This file intentionally has no external Python dependencies.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit


def _redactions() -> list[str]:
    values: set[str] = set()
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
        value = os.environ.get(name, "")
        if value:
            values.add(value)
            try:
                parsed = urlsplit(value)
                if parsed.password:
                    values.add(parsed.password)
                if parsed.username:
                    values.add(parsed.username + ":" + (parsed.password or ""))
            except ValueError:
                pass
    return sorted((x for x in values if x), key=len, reverse=True)


def redact(line: str, secrets: list[str]) -> str:
    for item in secrets:
        line = line.replace(item, "<redacted>")
    return re.sub(r"(?i)(https?://)[^\s/@]+:[^\s/@]+@", r"\1<redacted>@", line)


def kill_process_tree(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=15, check=False)
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        pass


def run_step(step: str, command: list[str], log_path: Path, timeout_seconds: int,
             input_file: Path | None = None, heartbeat_seconds: int = 15) -> int:
    if not command:
        raise ValueError("No subprocess command supplied")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    secrets = _redactions()
    events: queue.Queue[str | None] = queue.Queue()
    start = time.monotonic()

    def output(text: str, log) -> None:
        safe = redact(text, secrets)
        print(safe, flush=True)
        log.write(safe + "\n")
        log.flush()

    with log_path.open("a", encoding="utf-8") as log:
        output(f"[RPA] START: {step}", log)
        proc = subprocess.Popen(command, stdin=subprocess.PIPE if input_file else subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)

        def reader() -> None:
            assert proc.stdout is not None
            # Read binary lines to avoid local Windows codepage failures.
            for binary in iter(proc.stdout.readline, b""):
                events.put(binary.decode("utf-8", errors="replace").rstrip("\r\n"))
            events.put(None)

        worker = threading.Thread(target=reader, daemon=True)
        worker.start()
        if input_file:
            assert proc.stdin is not None
            try:
                proc.stdin.write(input_file.read_bytes())
                proc.stdin.close()
            except BrokenPipeError:
                pass
        eof = False
        last_activity = time.monotonic()
        timed_out = False
        while not eof or proc.poll() is None:
            now = time.monotonic()
            if now - start >= timeout_seconds:
                timed_out = True
                output(f"[RPA-E120] TIMEOUT at {step} after {timeout_seconds}s; terminating child process tree.", log)
                kill_process_tree(proc)
                break
            try:
                message = events.get(timeout=min(1, max(0.05, timeout_seconds - (now - start))))
                if message is None:
                    eof = True
                else:
                    output(message, log)
                    last_activity = time.monotonic()
            except queue.Empty:
                if time.monotonic() - last_activity >= heartbeat_seconds:
                    output(f"[RPA] {step} still running ({int(time.monotonic() - start)}s). "
                           "No recent output; press Ctrl+C to cancel, then rerun installer.", log)
                    last_activity = time.monotonic()
        # Drain any buffered lines after child exit.
        while not events.empty():
            message = events.get_nowait()
            if message is not None:
                output(message, log)
        if timed_out:
            return 124
        result = proc.wait()
        output(f"[RPA] END {step}: exit={result}, duration={int(time.monotonic() - start)}s", log)
        return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", required=True)
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--stdin-file", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command or args.timeout < 1:
        parser.error("Specify a command after -- and a positive --timeout")
    try:
        return run_step(args.step, command, args.log, args.timeout, args.stdin_file)
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[RPA-E121] {args.step}: cannot start or supervise command: {redact(str(exc), _redactions())}", file=sys.stderr)
        return 121


if __name__ == "__main__":
    raise SystemExit(main())
