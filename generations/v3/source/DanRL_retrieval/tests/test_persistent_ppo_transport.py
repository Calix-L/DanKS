import contextlib
import io
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from DanRL_retrieval.training.persistent_ppo_transport import maybe_run_client, read_message, write_message


class PersistentPPOTransportTest(unittest.TestCase):
    def test_missing_or_nonmatching_route_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            route = Path(tmp) / "route.json"
            with patch.dict("os.environ", {"DANRL_PERSISTENT_PPO_ROUTE": str(route)}):
                self.assertIsNone(maybe_run_client(["--output", str(Path(tmp) / "x.pt")]))

            route.write_text(
                json.dumps({"socket": str(Path(tmp) / "missing.sock"), "token": "x", "output_prefix": "/other/"}),
                encoding="utf-8",
            )
            with patch.dict("os.environ", {"DANRL_PERSISTENT_PPO_ROUTE": str(route)}):
                self.assertIsNone(maybe_run_client(["--output", str(Path(tmp) / "x.pt")]))

    def test_matching_route_round_trips_arguments_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            socket_path = root / "ppo.sock"
            route_path = root / "route.json"
            output_path = root / "checkpoints" / "run_update1.pt"
            output_path.parent.mkdir()
            token = "test-token"
            received = {}
            ready = threading.Event()

            def serve():
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                    server.bind(str(socket_path))
                    server.listen(1)
                    ready.set()
                    connection, _ = server.accept()
                    with connection:
                        received.update(read_message(connection))
                        write_message(connection, {"returncode": 0, "output": "trained\n"})

            thread = threading.Thread(target=serve, daemon=True)
            thread.start()
            self.assertTrue(ready.wait(2.0))
            route_path.write_text(
                json.dumps(
                    {
                        "socket": str(socket_path),
                        "token": token,
                        "output_prefix": str(output_path.parent / "run_"),
                    }
                ),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            argv = ["--output", str(output_path), "--epochs", "1"]
            with patch.dict("os.environ", {"DANRL_PERSISTENT_PPO_ROUTE": str(route_path)}):
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(maybe_run_client(argv), 0)
            thread.join(2.0)

            self.assertEqual(received["token"], token)
            self.assertEqual(received["argv"], argv)
            self.assertEqual(stdout.getvalue(), "trained\n")


if __name__ == "__main__":
    unittest.main()
