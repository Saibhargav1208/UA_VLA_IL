# UA-VLA-IL: QwenVLClient
# Lightweight HTTP client for the Qwen2-VL-2B server.
# Sends two VQA queries per action step:
#   1. Visual complexity of the current observation
#   2. Task precision requirement of the language instruction
#
# The server is launched separately:
#   python -m vla_cal.qwen_vl_server --port 12190

import re
import base64
from typing import Optional

import numpy as np

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

try:
    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from PIL import Image
    QWEN_AVAILABLE = True
except ImportError:
    QWEN_AVAILABLE = False


# ── Prompt templates ─────────────────────────────────────────────────────────

_COMPLEXITY_PROMPT = (
    "Look at this robot workspace image carefully. "
    "Rate the visual complexity of this scene on a scale from 0.0 to 1.0, where: "
    "0.0 = very simple (one object, clean background, high contrast), "
    "1.0 = very complex (many similar objects, cluttered, low contrast, distractors). "
    "Consider: number of objects, visual similarity between them, lighting, background clutter. "
    "Reply with a single decimal number only, e.g. 0.73"
)

_PRECISION_PROMPT = (
    "Given this robot manipulation task: \"{task}\". "
    "How precisely must the robot place or grasp the object? "
    "Rate from 0.0 to 1.0, where: "
    "0.0 = approximate placement is fine (e.g. 'put the block somewhere in the box'), "
    "1.0 = extremely precise placement required (e.g. 'insert the peg into the hole', 'stack block exactly on top'). "
    "Reply with a single decimal number only, e.g. 0.85"
)


def _extract_float(text: str) -> float:
    """Extract first float from model response and clamp to [0, 1]."""
    text = text.strip()
    matches = re.findall(r"\d+\.?\d*", text)
    if not matches:
        return 0.5  # neutral fallback
    val = float(matches[0])
    return float(np.clip(val, 0.0, 1.0))


def _ndarray_to_b64(image: np.ndarray) -> str:
    """Encode numpy RGB image as base64 PNG string for HTTP transport."""
    import io
    pil = Image.fromarray(image.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── Inline model (loaded on GPU, used by server) ─────────────────────────────

class QwenVLModel:
    """
    Qwen2-VL-2B-Instruct model for visual complexity and task precision estimation.
    Loaded once on GPU. Used by the server process.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: Optional[str] = None,
        max_new_tokens: int = 16,
    ) -> None:
        assert QWEN_AVAILABLE, (
            "Install transformers: pip install transformers qwen-vl-utils"
        )
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[QwenVLModel] Loading {model_name} on {device}...")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in device else torch.float32,
            device_map=device,
        )
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.device = device
        self.max_new_tokens = max_new_tokens
        print("[QwenVLModel] Ready.")

    def _run(self, image: np.ndarray, prompt: str) -> str:
        pil_img = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[chat_text], images=[pil_img],
            return_tensors="pt", padding=True
        ).to(self.device)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(generated, skip_special_tokens=True).strip()

    def estimate_complexity(self, image: np.ndarray) -> float:
        """Visual complexity of the scene. Returns float in [0, 1]."""
        response = self._run(image, _COMPLEXITY_PROMPT)
        return _extract_float(response)

    def estimate_precision(self, image: np.ndarray, task: str) -> float:
        """Task precision requirement. Returns float in [0, 1]."""
        prompt = _PRECISION_PROMPT.format(task=task)
        response = self._run(image, prompt)
        return _extract_float(response)


# ── HTTP client (used by the main process) ───────────────────────────────────

class QwenVLClient:
    """
    HTTP client for QwenVLModel server.
    Sends images and task descriptions to the Qwen2-VL-2B server.
    Falls back to neutral values (0.5) if server is unreachable.
    """

    def __init__(self, port: int = 12190, timeout: float = 5.0) -> None:
        self.base_url = f"http://localhost:{port}"
        self.complexity_url = f"{self.base_url}/complexity"
        self.precision_url = f"{self.base_url}/precision"
        self.timeout = timeout
        self._server_ok: Optional[bool] = None  # cached after first check

    def _check_server(self) -> bool:
        if self._server_ok is not None:
            return self._server_ok
        try:
            r = requests.get(f"{self.base_url}/health", timeout=2.0)
            self._server_ok = r.status_code == 200
        except Exception:
            self._server_ok = False
            print("[QwenVLClient] ⚠️  Server not reachable. Using neutral fallback (0.5).")
        return self._server_ok

    def estimate_complexity(self, image: np.ndarray) -> float:
        """
        Query Qwen2-VL for visual complexity score.
        Returns float in [0, 1]. Falls back to 0.5 if server is down.
        """
        if not REQUESTS_AVAILABLE or not self._check_server():
            return 0.5
        try:
            payload = {"image_b64": _ndarray_to_b64(image)}
            r = requests.post(self.complexity_url, json=payload, timeout=self.timeout)
            return float(r.json()["score"])
        except Exception as e:
            print(f"[QwenVLClient] complexity query failed: {e}")
            return 0.5

    def estimate_precision(self, image: np.ndarray, task: str) -> float:
        """
        Query Qwen2-VL for task precision requirement.
        Returns float in [0, 1]. Falls back to 0.5 if server is down.
        """
        if not REQUESTS_AVAILABLE or not self._check_server():
            return 0.5
        try:
            payload = {"image_b64": _ndarray_to_b64(image), "task": task}
            r = requests.post(self.precision_url, json=payload, timeout=self.timeout)
            return float(r.json()["score"])
        except Exception as e:
            print(f"[QwenVLClient] precision query failed: {e}")
            return 0.5
