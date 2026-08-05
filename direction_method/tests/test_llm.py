import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from llm import LLMClient, _cache_key, _fake_instance_for_schema, image_hash

IMG = np.zeros((8, 8, 3), dtype=np.uint8)


def test_fake_backend_returns_schema_conforming_json(tmp_path):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    schema = {"type": "object", "properties": {"verdict": {"type": "string", "enum": ["yes", "no"]}}, "required": ["verdict"]}
    result = client.call("does X hold?", IMG, response_schema=schema)
    parsed = json.loads(result.text)
    assert parsed["verdict"] == "yes"  # first enum value, deterministically


def test_fake_backend_free_text_when_no_schema(tmp_path):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    result = client.call("describe this", IMG)
    assert isinstance(result.text, str) and len(result.text) > 0


def test_cache_hit_on_identical_call(tmp_path):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    r1 = client.call("same prompt", IMG)
    assert r1.cached is False
    r2 = client.call("same prompt", IMG)
    assert r2.cached is True
    assert r2.text == r1.text


def test_cache_key_ignores_response_schema_per_spec_formula(tmp_path):
    """SPEC.md's cache key is literally sha256(model | prompt | image_hash) -- no schema
    component. So identical prompt+image+model hits cache even under a different schema. This
    is safe in practice because every real module uses one fixed schema per prompt *template*
    (self_check's two polarities are different prompt text, not the same text with two
    schemas) -- but it does mean callers must never reuse identical prompt text across two
    different expected response shapes.
    """
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    schema1 = {"type": "object", "properties": {"v": {"type": "string", "enum": ["a", "b"]}}, "required": ["v"]}
    schema2 = {"type": "object", "properties": {"v": {"type": "string", "enum": ["b", "a"]}}, "required": ["v"]}
    r1 = client.call("prompt", IMG, response_schema=schema1)
    r2 = client.call("prompt", IMG, response_schema=schema2)
    assert r1.cached is False
    assert r2.cached is True
    assert r2.text == r1.text  # schema2's own fake instance is never generated -- cache wins


def test_cache_miss_on_different_prompt_same_image(tmp_path):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    r1 = client.call("prompt one", IMG)
    r2 = client.call("prompt two", IMG)
    assert r1.cached is False and r2.cached is False


def test_cache_key_depends_on_model_prompt_and_image():
    h1 = _cache_key("model-a", "prompt", "hashA")
    h2 = _cache_key("model-b", "prompt", "hashA")
    h3 = _cache_key("model-a", "other-prompt", "hashA")
    h4 = _cache_key("model-a", "prompt", "hashB")
    assert len({h1, h2, h3, h4}) == 4  # every component changes the key


def test_image_hash_is_deterministic_and_content_sensitive():
    a = np.zeros((4, 4, 3), dtype=np.uint8)
    b = np.zeros((4, 4, 3), dtype=np.uint8)
    c = np.ones((4, 4, 3), dtype=np.uint8)
    assert image_hash(a) == image_hash(b)
    assert image_hash(a) != image_hash(c)


def test_image_hash_rejects_unsupported_type():
    try:
        image_hash("not an image")
        raise AssertionError("expected TypeError")
    except TypeError:
        pass


def test_corrupt_cache_file_is_treated_as_miss_not_crash(tmp_path):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    client.call("prompt", IMG)
    cache_files = list(client.cache_dir.glob("*.json"))
    assert len(cache_files) == 1
    cache_files[0].write_text("not valid json{{{")

    r2 = client.call("prompt", IMG)  # must not raise
    assert r2.cached is False


def test_cache_write_is_atomic_no_leftover_tmp_files(tmp_path):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    client.call("prompt", IMG)
    # client.cache_dir, not tmp_path: the fake backend namespaces its entries into a subdirectory
    # (see below), so globbing tmp_path itself would pass vacuously.
    tmp_files = list(client.cache_dir.glob("*.tmp"))
    assert tmp_files == []


def test_client_can_be_built_with_caching_off(tmp_path):
    """Measurement runs must query the model every time -- see LLMClient.__init__'s note on why
    temperature=0 does not make a cached run equivalent to a fresh one under vllm."""
    from llm import CALL_STATS, reset_call_stats

    reset_call_stats()
    client = LLMClient("m", backend="fake", cache_dir=tmp_path, use_cache=False)
    for _ in range(3):
        client.call("prompt", IMG)
    assert (CALL_STATS["hits"], CALL_STATS["misses"]) == (0, 3)  # never replayed
    assert list(client.cache_dir.glob("*.json")) == []  # and never written


def test_use_cache_env_var_turns_caching_off_process_wide(tmp_path, monkeypatch):
    # eval_model.py builds the questioner's LLMClient itself, so env is the only channel a driver
    # has to force real queries on that path.
    monkeypatch.setenv("VLM_USE_CACHE", "0")
    client = LLMClient("m", backend="fake", cache_dir=tmp_path)
    assert client.use_cache is False
    client.call("prompt", IMG)
    assert list(client.cache_dir.glob("*.json")) == []


def test_explicit_use_cache_false_at_call_site_still_wins(tmp_path):
    # context_parser's validation retry passes use_cache=False deliberately to bypass a hit.
    client = LLMClient("m", backend="fake", cache_dir=tmp_path)
    assert client.use_cache is True
    client.call("prompt", IMG)
    r = client.call("prompt", IMG, use_cache=False)
    assert r.cached is False


def test_call_stats_distinguish_a_cache_replay_from_a_real_run(tmp_path):
    """A run served entirely from cache is otherwise indistinguishable from one that queried the
    model -- the same blindness that let FakeVLM output pass as real in full_sweep_v1/v2, and a
    warm-cache replay pass as a fresh measurement. CALL_STATS makes it assertable."""
    from llm import CALL_STATS, reset_call_stats

    reset_call_stats()
    client = LLMClient("m", backend="fake", cache_dir=tmp_path)

    client.call("prompt", IMG)
    assert (CALL_STATS["hits"], CALL_STATS["misses"]) == (0, 1)  # cold: real query

    client.call("prompt", IMG)
    client.call("prompt", IMG)
    assert (CALL_STATS["hits"], CALL_STATS["misses"]) == (2, 1)  # warm: pure replay
    assert CALL_STATS["by_model"]["m"] == {"hits": 2, "misses": 1}

    # A second client on the same directory sees the same entries -- aggregation is process-wide,
    # which is what the official path needs (eval_model.py builds one questioner, and one
    # LLMClient, per episode inside an exec'd script).
    LLMClient("m", backend="fake", cache_dir=tmp_path).call("prompt", IMG)
    assert (CALL_STATS["hits"], CALL_STATS["misses"]) == (3, 1)
    reset_call_stats()
    assert (CALL_STATS["hits"], CALL_STATS["misses"]) == (0, 0)


def test_fake_backend_cache_is_namespaced_away_from_the_real_one(tmp_path):
    """Regression test for the contamination that produced full_sweep_v1/v2 (SPEC.md §13).

    The cache key is sha256(model | prompt | image_hash) -- no backend component -- so a FakeVLM
    entry sharing a directory with real ones is indistinguishable from a real result, and a later
    "real" run replays the placeholder instead of calling the model. Separate directories make
    that collision impossible: identical model+prompt+image, different backend, different file.
    """
    fake = LLMClient("m", backend="fake", cache_dir=tmp_path)
    fake.call("prompt", IMG)

    assert fake.cache_dir == Path(tmp_path) / "_fake"
    assert len(list(fake.cache_dir.glob("*.json"))) == 1
    # Nothing written where a real-backend client would look.
    assert list(Path(tmp_path).glob("*.json")) == []


def test_latency_accumulates_across_calls(tmp_path):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    assert client.time_required == 0.0
    client.call("p1", IMG, use_cache=False)
    t1 = client.time_required
    client.call("p2", IMG, use_cache=False)
    assert client.time_required >= t1


def test_cached_call_does_not_add_latency(tmp_path):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    client.call("prompt", IMG, use_cache=True)
    t_after_first = client.time_required
    client.call("prompt", IMG, use_cache=True)  # cache hit
    assert client.time_required == t_after_first


def test_schema_filler_enum_takes_first_value():
    assert _fake_instance_for_schema({"type": "string", "enum": ["contradicts", "consistent", "cant_tell"]}) == "contradicts"


def test_schema_filler_handles_nested_object_and_array():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"label": {"type": "string"}, "score": {"type": "number"}},
                    "required": ["label", "score"],
                },
            },
            "count": {"type": "integer"},
        },
        "required": ["items", "count"],
    }
    instance = _fake_instance_for_schema(schema)
    assert isinstance(instance["items"], list) and len(instance["items"]) == 1
    assert "label" in instance["items"][0] and "score" in instance["items"][0]
    assert isinstance(instance["count"], int)


def test_schema_filler_respects_min_items_for_primitive_arrays():
    # zone_gen's bbox_2d is `{"type": "array", "items": {"type": "integer"}, "minItems": 4,
    # "maxItems": 4}` -- the filler must produce all 4 elements, not just one, or the caller's
    # `x0, y0, x1, y1 = bbox_2d` unpacking crashes under FakeVLM.
    schema = {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}
    assert _fake_instance_for_schema(schema) == [0, 0, 0, 0]


def test_no_image_calls_do_not_crash(tmp_path):
    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path)
    result = client.call("text-only prompt, no image")
    assert isinstance(result.text, str)


def test_default_backend_is_vllm_not_fake(tmp_path, monkeypatch):
    monkeypatch.delenv("VLM_BACKEND", raising=False)
    from llm import FakeVLMBackend

    client = LLMClient("fake-model-name-unused", cache_dir=tmp_path)
    assert not isinstance(client._backend, FakeVLMBackend)


def test_env_var_selects_fake_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("VLM_BACKEND", "fake")
    from llm import FakeVLMBackend

    client = LLMClient("fake-model-name-unused", cache_dir=tmp_path)
    assert isinstance(client._backend, FakeVLMBackend)


def test_concurrent_calls_sharing_a_cache_key_do_not_crash(tmp_path):
    """Regression test: two threads racing _store_cache for the identical cache key (same
    prompt+image -- realistic under the full-sweep driver's ThreadPoolExecutor, e.g. two
    episode-runs asking the exact same self_check question against the same target image) used
    to collide on a tmp path keyed only by os.getpid(), which every thread in one process shares.
    The loser's tmp_path.replace() then raised FileNotFoundError because the winner had already
    renamed the shared tmp file away. Fixed by keying the tmp suffix on threading.get_ident() too.
    """
    import threading

    client = LLMClient("fake-model", backend="fake", cache_dir=tmp_path, timeout_s=5.0)
    errors = []

    def worker():
        try:
            client.call("identical prompt for every thread", IMG, use_cache=True)
        except Exception as e:  # noqa: BLE001 -- the test's whole point is that this must not happen
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
