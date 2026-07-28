"""Agent-as-VLM bridge: lets Claude itself answer every VLM call (oracle included) by actually
reading the prompt and image, instead of a real vllm server or FakeVLM's canned responses.

Not a new LLMClient backend registered in llm.py -- this is a standalone class whose `generate`
method gets monkeypatched onto a normal LLMClient's `_backend`, exactly like every test in this
package already does (`client._backend.generate = ...`). Zero changes needed anywhere else:
self_check.py / zone_gen.py / context_parser.py / checklist_update.py / questioner.py /
oracle_stub.py all just call `llm_client.call(...)`, which calls `self._backend.generate(...)`,
completely backend-agnostic.

Protocol, one request/response file pair per call:
  - `generate()` saves the image (if any) to `queue/images/<id>.png`, writes
    `queue/requests/<id>.json` with the full prompt text, the image path, and the response
    schema (None for free-text calls -- currently only the oracle), then prints
    "AGENT_REQUEST_READY <id>" to stdout (for a Monitor watching this process) and blocks,
    polling for `queue/responses/<id>.json` to appear.
  - The agent answering (Claude, in the driving conversation) reads the request + image, decides
    the answer, and writes `queue/responses/<id>.json` as `{"text": "<answer or JSON string>"}`.
    For schema-constrained calls, "text" must be a JSON string matching the schema (this is
    exactly what a real backend's raw completion text would be, before the caller's own
    `json.loads`). For the oracle's free-text calls (response_schema is None), "text" is just the
    plain-language answer.
"""

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image


def _save_image(image, path: Path) -> None:
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(path)
    elif isinstance(image, Image.Image):
        image.save(path)
    else:
        raise TypeError(f"Unsupported image type for the agent bridge: {type(image)}")


class AgentBridgeBackend:
    def __init__(self, queue_dir, *, poll_interval_s: float = 1.0, max_wait_s: float = 7200.0):
        self.queue_dir = Path(queue_dir)
        self.requests_dir = self.queue_dir / "requests"
        self.responses_dir = self.queue_dir / "responses"
        self.images_dir = self.queue_dir / "images"
        for d in (self.requests_dir, self.responses_dir, self.images_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.poll_interval_s = poll_interval_s
        self.max_wait_s = max_wait_s  # generous: a real conversational turn, not a real model call
        self._counter = 0

    def generate(self, prompt: str, image, response_schema: dict | None, timeout_s: float) -> str:
        self._counter += 1
        req_id = f"{self._counter:05d}"

        image_path = None
        if image is not None:
            image_path = self.images_dir / f"{req_id}.png"
            _save_image(image, image_path)

        request = {
            "id": req_id,
            "prompt": prompt,
            "image_path": str(image_path) if image_path else None,
            "response_schema": response_schema,
        }
        (self.requests_dir / f"{req_id}.json").write_text(json.dumps(request, indent=2))
        print(f"AGENT_REQUEST_READY {req_id}", flush=True)

        response_path = self.responses_dir / f"{req_id}.json"
        waited = 0.0
        while not response_path.exists():
            time.sleep(self.poll_interval_s)
            waited += self.poll_interval_s
            if waited > self.max_wait_s:
                raise TimeoutError(f"No agent response for request {req_id} after {self.max_wait_s}s")

        response = json.loads(response_path.read_text())
        return response["text"]
