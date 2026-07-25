"""OpenAI-compatible vllm client wrapper: caching, timing, timeout/retry (spec §7).

No other module talks to the vllm server directly — everything (extract.py, adjudicate.py,
parse.py's future LLM path) goes through `LLMClient.call()`.
"""

import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))  # utils.py / Oracle.py / env.py live at repo root

from utils import ClientBasedLLM  # noqa: E402


@dataclass
class LLMResult:
    text: str
    logprobs: list | None
    latency_s: float
    cached: bool


def _image_hash(image) -> str:
    if isinstance(image, np.ndarray):
        payload = image.tobytes()
    elif isinstance(image, Image.Image):
        payload = image.tobytes()
    else:
        raise TypeError(f"Unsupported image type for hashing: {type(image)}")
    return hashlib.sha256(payload).hexdigest()


def _cache_key(model_id: str, prompt: str, image_hash: str | None, temperature: float) -> str:
    blob = f"{model_id}|{prompt}|{image_hash or ''}|{temperature}".encode()
    return hashlib.sha256(blob).hexdigest()


class LLMClient:
    """Thin, cached, time-accounted wrapper over `ClientBasedLLM`.

    `time_required` accumulates wall-clock spent on *actual* (non-cached) calls, so it can be
    added straight into `GraphQuestioner.time_required` (spec §0.5 — eval_model.py reads that
    attribute for logging).
    """

    def __init__(
        self,
        model_id: str,
        *,
        port: int = 8000,
        temperature: float = 0.0,
        cache_dir: str | Path = "artifacts/cache",
        timeout_s: float = 20.0,
        retries: int = 1,
    ):
        self.model_id = model_id
        self.temperature = temperature
        self.timeout_s = timeout_s
        self.retries = retries
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.time_required = 0.0
        # attrs strips the leading underscore from `_port`/`_url` for the __init__ kwarg name.
        self._client = ClientBasedLLM(model_id=model_id, temperature=temperature, port=port)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cached(self, key: str) -> LLMResult | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        with open(path) as f:
            payload = json.load(f)
        return LLMResult(text=payload["text"], logprobs=payload.get("logprobs"), latency_s=0.0, cached=True)

    def _store_cache(self, key: str, text: str, logprobs) -> None:
        with open(self._cache_path(key), "w") as f:
            json.dump({"text": text, "logprobs": logprobs}, f)

    def call(
        self,
        prompt: str,
        image=None,
        *,
        temperature: float | None = None,
        want_logprobs: bool = False,
        top_logprobs: int = 5,
        use_cache: bool = True,
    ) -> LLMResult:
        """Returns (text, logprobs, latency_s, cached) via LLMResult. On repeated failure,
        degrades by raising `LLMCallFailed` — callers must catch this and fall back per spec §7
        ("on failure, degrade: skip extraction -> treat all slots unknown -> adjudicate"). This
        wrapper never retries past `self.retries`, so one flaky call can't consume the episode
        clock the way an unbounded retry loop would.
        """
        temp = self.temperature if temperature is None else temperature
        image_hash = _image_hash(image) if image is not None else None
        key = _cache_key(self.model_id, prompt, image_hash, temp)

        if use_cache:
            cached = self._load_cached(key)
            if cached is not None:
                return cached

        kwargs = {}
        if want_logprobs:
            kwargs["logprobs"] = True
            kwargs["top_logprobs"] = top_logprobs

        last_err = None
        for attempt in range(self.retries + 1):
            start = time.monotonic()
            try:
                result = self._client.ask(
                    prompt=prompt,
                    images=[image] if image is not None else None,
                    timeout=self.timeout_s,
                    **kwargs,
                )
                latency = time.monotonic() - start
                self.time_required += latency
                text, logprobs = (result if want_logprobs else (result, None))
                if use_cache:
                    self._store_cache(key, text, logprobs)
                return LLMResult(text=text, logprobs=logprobs, latency_s=latency, cached=False)
            except Exception as e:  # noqa: BLE001 — vllm/network errors are heterogeneous; degrade uniformly
                self.time_required += time.monotonic() - start
                last_err = e
        raise LLMCallFailed(f"LLM call failed after {self.retries + 1} attempt(s): {last_err}") from last_err


class LLMCallFailed(RuntimeError):
    pass
