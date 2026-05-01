"""
UA-VLA-IL: Qwen2-VL-2B server
Hosts the visual complexity and task precision estimators.

Usage:
    python -m vla_cal.qwen_vl_server --port 12190 --model Qwen/Qwen2-VL-2B-Instruct

Endpoints:
    GET  /health       → {"status": "ok"}
    POST /complexity   → {"score": float}   body: {"image_b64": str}
    POST /precision    → {"score": float}   body: {"image_b64": str, "task": str}
"""

import argparse
import base64
import io
import json
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np
from PIL import Image

from vla_cal.qwen_vl_client import QwenVLModel


_MODEL: QwenVLModel = None  # set after loading


def _decode_image(image_b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(image_b64)
    pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return np.array(pil)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # suppress per-request logs

    def _send_json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            payload = self._read_json()

            if self.path == "/complexity":
                image = _decode_image(payload["image_b64"])
                score = _MODEL.estimate_complexity(image)
                self._send_json({"score": score})

            elif self.path == "/precision":
                image = _decode_image(payload["image_b64"])
                task = payload.get("task", "")
                score = _MODEL.estimate_precision(image, task)
                self._send_json({"score": score})

            else:
                self._send_json({"error": f"unknown endpoint: {self.path}"}, 404)

        except Exception as e:
            self._send_json({"error": str(e)}, 500)


def main():
    global _MODEL

    parser = argparse.ArgumentParser(description="Qwen2-VL-2B server for UA-VLA-IL")
    parser.add_argument("--port", type=int, default=12190)
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    print(f"[Server] Loading {args.model}...")
    _MODEL = QwenVLModel(model_name=args.model, device=args.device)

    server = HTTPServer(("localhost", args.port), _Handler)
    print(f"[Server] Listening on port {args.port}. Ctrl-C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[Server] Shutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
