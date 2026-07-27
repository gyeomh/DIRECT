"""One VLM client for all four method modules + the local oracle stand-in (SPEC.md §9 step 2).

Backend-agnostic by design: `VLM_BACKEND=fake` (or `LLMClient(..., backend="fake")`) swaps in
`FakeVLMBackend`, which returns schema-conforming canned JSON (or a placeholder string for plain
text calls) with the exact same call signature as the real backend — the whole pipeline can be
exercised end to end without a server. Default backend is the real one (`vllm`), so nothing here
needs to change tomorrow; only the environment (a live server on the configured port) does.

Image-first, byte-identical prefix: every multimodal call places the image block before the text
block (via `ClientBasedLLM._image_text_chat`'s existing message shape — `[image_url, text]`), so
two calls sharing an image share a byte-identical prompt prefix regardless of what text follows.
That's what lets vllm's automatic prefix caching reuse the image's KV cache across calls
(SPEC §8) — `check_prefix_caching()` at the bottom measures whether it actually does, but is not
invoked here; there's no server to check against until tomorrow.
"""

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils import ClientBasedLLM  # noqa: E402


def image_hash(image) -> str:
    if isinstance(image, np.ndarray):
        payload = image.tobytes()
    elif isinstance(image, Image.Image):
        payload = image.tobytes()
    else:
        raise TypeError(f"Unsupported image type for hashing: {type(image)}")
    return hashlib.sha256(payload).hexdigest()


def _cache_key(model_id: str, prompt: str, img_hash: str | None) -> str:
    # Exactly sha256(model | prompt | image_hash) per SPEC.md §9 step 2 — deliberately no
    # temperature component (every call here runs at temperature=0.0; see VLLMBackend).
    blob = f"{model_id}|{prompt}|{img_hash or ''}".encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass
class LLMResult:
    text: str
    latency_s: float
    cached: bool


def _fake_instance_for_schema(schema: dict):
    """Walks a JSON schema and produces a minimal, structurally valid instance: enums take their
    first listed value, objects fill every property, arrays get one example item. This is what
    lets FakeVLMBackend return a plausible response for *any* module's schema without bespoke
    per-module canned text — self_check's {verdict: enum[...]} and zone_gen's bbox-list schema
    both go through the same code path.
    """
    if "enum" in schema:
        return schema["enum"][0]
    t = schema.get("type")
    if t == "object":
        props = schema.get("properties", {})
        return {k: _fake_instance_for_schema(v) for k, v in props.items()}
    if t == "array":
        item_schema = schema.get("items", {"type": "string"})
        return [_fake_instance_for_schema(item_schema)]
    if t == "integer":
        return 0
    if t == "number":
        return 0.0
    if t == "boolean":
        return False
    return "fake"


class FakeVLMBackend:
    """Canned responses matching each expected output schema. Same call signature as
    VLLMBackend — LLMClient doesn't know or care which backend it's holding.
    """

    def generate(self, prompt: str, image, response_schema: dict | None, timeout_s: float) -> str:
        if response_schema is not None:
            return json.dumps(_fake_instance_for_schema(response_schema))
        return "This is a fake VLM free-text response for offline testing."


class VLLMBackend:
    """Real backend: OpenAI-compatible vllm server via the repo's own ClientBasedLLM.

    TODO(tomorrow, needs a live server to confirm): `extra_body={"guided_json": schema}` is the
    classic vllm guided-decoding parameter shape, matching SPEC.md §8's own terminology
    ("guided_json for zone_gen and self_check outputs"). Some vllm versions instead expect the
    OpenAI-native `response_format={"type": "json_schema", "json_schema": {...}}`. Verify against
    whichever vllm version actually gets served tomorrow — if `extra_body` errors or is silently
    ignored (model ignores the schema), switch to `response_format`.
    """

    def __init__(self, model_id: str, *, port: int = 8000, temperature: float = 0.0):
        self.model_id = model_id
        self.temperature = temperature
        self._client = ClientBasedLLM(model_id=model_id, temperature=temperature, port=port)

    def generate(self, prompt: str, image, response_schema: dict | None, timeout_s: float) -> str:
        kwargs = {"timeout": timeout_s}
        if response_schema is not None:
            kwargs["extra_body"] = {"guided_json": response_schema}
        return self._client.ask(prompt=prompt, images=[image] if image is not None else None, **kwargs)

    def stream_with_ttft(self, prompt: str, image, response_schema: dict | None):
        """Low-level streaming call exposing time-to-first-token, used only by
        check_prefix_caching(). The normal .generate() path (via ClientBasedLLM.ask) buffers the
        full response internally and discards per-chunk timing, so TTFT — the best available
        proxy for prefill time through a standard OpenAI-compatible API — isn't otherwise
        reachable without vllm's own Prometheus metrics endpoint.
        """
        from utils import encode_image_b64  # repo util, image -> base64 for the raw message

        if isinstance(image, np.ndarray):
            pil_image = Image.fromarray(image)
            image_format = "png"
        else:
            pil_image = image
            image_format = image.format or "png"
        image_b64 = encode_image_b64(pil_image, image_format)

        kwargs = {}
        if response_schema is not None:
            kwargs["extra_body"] = {"guided_json": response_schema}

        start = time.monotonic()
        ttft = None
        parts = []
        stream = self._client._client.chat.completions.create(  # noqa: SLF001 — intentional, see docstring
            model=self.model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/{image_format.lower()};base64,{image_b64}"}},
                    {"type": "text", "text": prompt},
                ],
            }],
            temperature=self.temperature,
            stream=True,
            **kwargs,
        )
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                if ttft is None:
                    ttft = time.monotonic() - start
                parts.append(token)
        total = time.monotonic() - start
        return "".join(parts), ttft, total


class LLMClient:
    def __init__(
        self,
        model_id: str,
        *,
        backend: str | None = None,
        cache_dir: str | Path = "artifacts/cache",
        timeout_s: float = 20.0,
        port: int = 8000,
        temperature: float = 0.0,
    ):
        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.time_required = 0.0

        self.backend_name = backend or os.environ.get("VLM_BACKEND", "vllm")
        if self.backend_name == "fake":
            self._backend = FakeVLMBackend()
        else:
            self._backend = VLLMBackend(model_id, port=port, temperature=temperature)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cached(self, key: str) -> LLMResult | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with open(path) as f:
                payload = json.load(f)
        except json.JSONDecodeError:
            return None  # corrupt/truncated cache file reads as a miss, not a crash
        return LLMResult(text=payload["text"], latency_s=0.0, cached=True)

    def _store_cache(self, key: str, text: str) -> None:
        # Write-then-rename: a crash partway through json.dump must not leave a truncated file
        # at the real cache path (it would read back as corrupt on every future call for this key).
        final_path = self._cache_path(key)
        tmp_path = final_path.with_suffix(f".{os.getpid()}.tmp")
        with open(tmp_path, "w") as f:
            json.dump({"text": text}, f)
        tmp_path.replace(final_path)

    def call(
        self,
        prompt: str,
        image=None,
        *,
        response_schema: dict | None = None,
        use_cache: bool = True,
    ) -> LLMResult:
        img_hash = image_hash(image) if image is not None else None
        key = _cache_key(self.model_id, prompt, img_hash)

        if use_cache:
            cached = self._load_cached(key)
            if cached is not None:
                return cached

        start = time.monotonic()
        text = self._backend.generate(prompt, image, response_schema, self.timeout_s)
        latency = time.monotonic() - start
        self.time_required += latency

        if use_cache:
            self._store_cache(key, text)
        return LLMResult(text=text, latency_s=latency, cached=False)


def check_prefix_caching(client: LLMClient, image, suffix_a: str, suffix_b: str) -> tuple[float, float]:
    """SPEC.md §8: 'Confirm prefix caching actually engages.' Same image, two different text
    suffixes, both uncached — reports time-to-first-token for each. If prefix caching is
    working, the image's KV cache is computed once (on whichever call runs first) and reused by
    the second, so the second call's TTFT should be noticeably lower than the first's (its
    prefill is cheaper — text-only suffix, not text+image).

    Included but NOT invoked anywhere in this module or in tests: there is no vllm server to
    check against until tomorrow. Run manually once one is up:

        from llm import LLMClient, check_prefix_caching
        client = LLMClient("Qwen/Qwen3-VL-30B-A3B-Instruct")
        check_prefix_caching(client, some_image, "Describe the color.", "Describe the shape.")
    """
    if client.backend_name != "vllm" or not isinstance(client._backend, VLLMBackend):
        raise RuntimeError("check_prefix_caching needs the real vllm backend, not FakeVLM.")

    _, ttft_a, total_a = client._backend.stream_with_ttft(suffix_a, image, None)
    _, ttft_b, total_b = client._backend.stream_with_ttft(suffix_b, image, None)

    print(f"suffix_a: TTFT={ttft_a}, total={total_a:.3f}s  ({suffix_a!r})")
    print(f"suffix_b: TTFT={ttft_b}, total={total_b:.3f}s  ({suffix_b!r})")
    print("If prefix caching engaged, suffix_b's TTFT should be markedly lower than suffix_a's.")
    return ttft_a, ttft_b
