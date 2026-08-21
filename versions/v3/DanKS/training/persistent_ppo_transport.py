from __future__ import annotations

import json
import os
import socket
import sys
from pathlib import Path
from typing import Any


DEFAULT_ROUTE_FILE = Path("/dev/shm/danks_persistent_ppo_route.json")


def _argument_value(argv: list[str], name: str) -> str | None:
    try:
        index = argv.index(name)
    except ValueError:
        return None
    return argv[index + 1] if index + 1 < len(argv) else None


def read_message(connection: socket.socket) -> dict[str, Any]:
    chunks: list[bytes] = []
    while True:
        chunk = connection.recv(65536)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break
    payload = b"".join(chunks).split(b"\n", 1)[0]
    if not payload:
        raise RuntimeError("persistent PPO connection closed without a message")
    return json.loads(payload.decode("utf-8"))


def write_message(connection: socket.socket, payload: dict[str, Any]) -> None:
    connection.sendall(json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n")


def run_persistent_client(
    argv: list[str],
    *,
    route_file: Path | None = None,
    cwd: str | None = None,
) -> tuple[int, str] | None:
    """Send one request without spawning a forwarding Python process."""

    argv = list(argv)
    route_file = route_file or Path(
        os.environ.get("DANKS_PERSISTENT_PPO_ROUTE", str(DEFAULT_ROUTE_FILE))
    )
    try:
        route = json.loads(route_file.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

    output = _argument_value(argv, "--output")
    output_prefix = str(route.get("output_prefix", ""))
    if not output or not output_prefix or not str(Path(output).resolve()).startswith(output_prefix):
        return None

    socket_path = str(route.get("socket", ""))
    token = str(route.get("token", ""))
    if not socket_path or not token:
        return None

    request_sent = False
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
            connection.settimeout(5.0)
            connection.connect(socket_path)
            write_message(
                connection,
                {
                    "token": token,
                    "argv": argv,
                    "cwd": cwd or os.getcwd(),
                },
            )
            request_sent = True
            connection.shutdown(socket.SHUT_WR)
            connection.settimeout(300.0)
            response = read_message(connection)
    except (OSError, RuntimeError, json.JSONDecodeError) as exc:
        if not request_sent:
            return None
        return 1, f"persistent PPO request failed after dispatch: {exc!r}\n"

    output_text = str(response.get("output", ""))
    return int(response.get("returncode", 1)), output_text


def maybe_run_client(argv: list[str] | None = None) -> int | None:
    """Route a matching train_ppo invocation to a persistent worker.

    Returning ``None`` means no active matching route and lets the caller run
    the ordinary local training path. Once a request is sent, failures return
    non-zero rather than replaying a potentially completed PPO update.
    """

    result = run_persistent_client(list(sys.argv[1:] if argv is None else argv))
    if result is None:
        return None
    returncode, output_text = result
    if output_text:
        sys.stdout.write(output_text)
        if not output_text.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()
    return returncode
