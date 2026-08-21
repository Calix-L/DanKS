#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import secrets
import signal
import socket
import sys
import time
import traceback
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from DanRL_retrieval.training import train_ppo  # noqa: E402
from DanRL_retrieval.training.persistent_ppo_transport import read_message, write_message  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Persistent process wrapper for repeated train_ppo jobs.")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--route-file", required=True)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    args = parse_args()
    socket_path = Path(args.socket)
    route_file = Path(args.route_file)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    route_file.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    token = secrets.token_hex(16)
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(str(socket_path))
        server.listen(4)
        server.settimeout(1.0)
        atomic_json(
            route_file,
            {
                "socket": str(socket_path),
                "token": token,
                "output_prefix": str(Path(args.output_prefix).resolve()),
                "pid": os.getpid(),
            },
        )
        print(f"persistent_ppo_ready socket={socket_path} route={route_file}", flush=True)
        job_id = 0
        while not stopping:
            try:
                connection, _address = server.accept()
            except socket.timeout:
                continue
            with connection:
                job_id += 1
                job_start = time.perf_counter()
                output = io.StringIO()
                returncode = 0
                requested_output = ""
                old_argv = sys.argv
                old_cwd = os.getcwd()
                try:
                    request = read_message(connection)
                    if request.get("token") != token:
                        raise PermissionError("persistent PPO token mismatch")
                    argv = request.get("argv")
                    if not isinstance(argv, list) or not all(isinstance(value, str) for value in argv):
                        raise ValueError("persistent PPO argv must be a string list")
                    if "--output" in argv and argv.index("--output") + 1 < len(argv):
                        requested_output = str(Path(argv[argv.index("--output") + 1]).resolve())
                    if not requested_output or not requested_output.startswith(str(Path(args.output_prefix).resolve())):
                        raise PermissionError("persistent PPO output is outside the configured prefix")
                    os.chdir(str(request.get("cwd") or ROOT))
                    sys.argv = [str(Path(train_ppo.__file__).resolve()), *argv]
                    with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                        train_ppo.main()
                except BaseException:
                    returncode = 1
                    traceback.print_exc(file=output)
                finally:
                    sys.argv = old_argv
                    os.chdir(old_cwd)
                elapsed = time.perf_counter() - job_start
                output.write(f"persistent_ppo_job={job_id} elapsed_sec={elapsed:.3f}\n")
                try:
                    write_message(connection, {"returncode": returncode, "output": output.getvalue()})
                except OSError:
                    pass
                print(
                    f"persistent_ppo_job_done job={job_id} rc={returncode} elapsed_sec={elapsed:.3f} "
                    f"output={requested_output or 'unknown'}",
                    flush=True,
                )

    try:
        current = json.loads(route_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        current = {}
    if current.get("token") == token:
        route_file.unlink(missing_ok=True)
    socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
