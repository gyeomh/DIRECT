"""Run DirectionMethodQuestioner through the REAL, upstream `eval_model.py` __main__ block.

Every other driver in this folder (`run_full_sweep.py`, `run_full_loop_dry_run.py`,
`run_agent_verification.py`) reimplements the candidate loop against `env.QAEnv` directly. That
is deliberate — it buys resumability, threading, and rich per-run logs the upstream script has no
way to produce. But it also means the exact code path the official evaluation uses has never
actually executed this questioner, so anything upstream does that our loop doesn't (the bare
`except:` around construction, `already_done_ids`, the gzip log write, the `add_answer` call
order, `reset_time()` before the loop) is untested.

This script closes that gap. It does NOT reimplement the loop: it loads upstream's own source,
applies only the ENV.md §6 patches (in memory — the file on disk stays byte-identical to
upstream, verified by `git diff`), substitutes the questioner import, and `exec`s it. Same
mechanism `verify_env.py` already uses for `TrivialQuestioner`; this points it at the real thing.

Oracle choice (`--oracle`):
  upstream  upstream's own construction block, untouched — Gemini, or a local VLM via
            ORACLE_MODEL_ID + --local 1. This is the only setting that matches an official run.
  stub      LocalOracleStandIn on our own LLMClient (SPEC.md's development stand-in). Costs no
            API credits and needs no second server.
  mock      WorkingMockOracle — one fixed answer string. Wiring test only; the answers are
            meaningless, so accuracy from this setting means nothing.

Questioner backend: set VLM_BACKEND=fake to drive the whole thing off FakeVLM with no GPU and no
server at all. That combination (`--oracle mock` + `VLM_BACKEND=fake`) is a pure plumbing check —
it proves the official path constructs, runs, and logs this questioner without touching a model.

GPU note: any real run of this must have its vllm server pinned to GPU 2 (project constraint).
This script never spawns a server itself; it only connects to one via VLM_PORT.
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DIRECTION_ROOT = HERE.parent
REPO_ROOT = DIRECTION_ROOT.parent
sys.path.insert(0, str(DIRECTION_ROOT))
sys.path.insert(0, str(REPO_ROOT))

from patches.apply_patches import (  # noqa: E402
    WorkingMockOracle,
    ensure_episodes_symlink,
    load_patched_eval_model_source,
    substitute_questioner_import,
)

# Upstream's oracle-construction block, verbatim (eval_model.py:91-100). Substituted only for the
# non-`upstream` --oracle settings; an exact-match failure means eval_model.py changed and ENV.md
# needs re-verifying, so it raises rather than silently running with the wrong oracle.
_ORACLE_BLOCK = '''    if not local:
        oracle_client = GeminiLLM(model_id=ORACLE_MODEL_ID, temperature=1e-6)
        print(f"[INFO] Using oracle model: {ORACLE_MODEL_ID}")
    else:
        oracle_model_id = os.environ["ORACLE_MODEL_ID"]
        oracle_client = ClientBasedLLM(model_id=oracle_model_id)
        ## Or you can use your oracle here
        # oracle_client = YourOracle
        print(f"[INFO] Using oracle model: {oracle_model_id}")
        # TODO You can also use your oracle here'''

_INJECTED_ORACLE_BLOCK = "    oracle_client = _injected_oracle  # substituted by run_official_eval.py --oracle"

_QUESTIONER_IMPORT = "from questioner import DirectionMethodQuestioner as YourQuestioner"


def build_oracle(kind: str):
    if kind == "upstream":
        return None
    if kind == "mock":
        return WorkingMockOracle()
    if kind == "stub":
        from llm import LLMClient
        from oracle_stub import LocalOracleStandIn

        from patches.apply_patches import fix_answer_prompt_typo

        fix_answer_prompt_typo()
        from questioner import DEFAULT_DISABLE_THINKING, DEFAULT_MODEL_ID

        return LocalOracleStandIn(
            LLMClient(
                os.environ.get("ORACLE_MODEL_ID", DEFAULT_MODEL_ID),
                disable_thinking=DEFAULT_DISABLE_THINKING,
            )
        )
    raise ValueError(f"unknown --oracle {kind!r}")


def preflight_vllm() -> None:
    """Fail loudly and accurately if the vllm server isn't reachable.

    eval_model.py:153-158 wraps questioner construction in a BARE `except:` that re-raises a
    fixed NotImplementedError("Insert here your Questioner class"). Every real construction
    failure -- including `openai.APIConnectionError` from a server that is down, or on a different
    port -- is swallowed and reported as though the questioner were never wired up. Confirmed by
    running this script against a dead port. Checking here, before exec, keeps that misdirection
    from costing an hour of debugging on the GPU box.
    """
    import json as _json
    import urllib.error
    import urllib.request

    port = int(os.environ.get("VLM_PORT", 8000))
    url = f"http://127.0.0.1:{port}/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:  # noqa: S310 -- fixed localhost URL
            served = [m.get("id") for m in _json.load(r).get("data", [])]
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise SystemExit(
            f"[preflight] No OpenAI-compatible server answering {url} ({e}).\n"
            f"            Start vllm (pinned to GPU 2 -- project constraint) and/or set VLM_PORT.\n"
            f"            Without this, eval_model.py's bare `except:` would report the real error\n"
            f"            as 'Insert here your Questioner class', which is not what went wrong."
        ) from e
    if not served:
        raise SystemExit(f"[preflight] {url} answered but serves no models.")

    # The served model must be the one the questioner will actually ask for. Printing it is not
    # enough: on 2026-08-05 a full 55-minute sweep ran to completion on the superseded baseline
    # because the served model was reported, read, and not compared against anything.
    from questioner import DEFAULT_MODEL_ID

    wanted = DEFAULT_MODEL_ID
    if wanted not in served:
        raise SystemExit(
            f"[preflight] MODEL MISMATCH -- refusing to run.\n"
            f"            questioner will request : {wanted}\n"
            f"            server on port {port} serves: {served}\n"
            f"            Every call would 404 or silently answer from the wrong model. Start the\n"
            f"            right server, or override with VLM_MODEL_ID if the mismatch is intended."
        )

    # /v1/models is served by the API process and answers even when the ENGINE is dead -- confirmed
    # the hard way: Qwen3.6's gated-delta-rule path JIT-compiles through flashinfer on the first
    # real prefill, and with `ninja` missing from the env the engine died there while /v1/models
    # kept returning 200. The sweep then ran for minutes against a corpse. So actually generate:
    # a real completion is the only thing that exercises the path that broke.
    body = _json.dumps({
        "model": served[0],
        "messages": [{"role": "user", "content": "Reply with the single word: ok"}],
        "max_tokens": 8,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:  # noqa: S310 -- fixed localhost URL
            out = _json.load(r)["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001 -- any failure here means the engine cannot generate
        raise SystemExit(
            f"[preflight] {served[0]} is listed but FAILED to generate ({e!r}).\n"
            f"            The engine is down even though /v1/models answers. Check the server log."
        ) from e
    print(f"[preflight] vllm on port {port} serving {served[0]}, generation OK: {out.strip()[:40]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("start_idx", type=int)
    ap.add_argument("end_idx", type=int)
    ap.add_argument("--description-type", default="category")
    ap.add_argument("--oracle", default="stub", choices=["upstream", "stub", "mock"])
    ap.add_argument("--local", type=int, default=0, help="passed through to eval_model.py; only meaningful with --oracle upstream")
    args = ap.parse_args()

    # eval_model.py runs with cwd=REPO_ROOT and constructs the questioner as `YourQuestioner(info)`,
    # so LLMClient's relative default cache dir would resolve to a fresh, cold cache at the repo
    # root rather than the populated one the sweeps built. Pin it unless the caller already has.
    os.environ.setdefault("VLM_CACHE_DIR", str(DIRECTION_ROOT / "artifacts" / "cache"))

    # patches/README.md #5: eval_model.py's final gzip write does not create parent directories,
    # so a full run crashes at the very last step without this.
    (REPO_ROOT / "results").mkdir(exist_ok=True)
    ensure_episodes_symlink(REPO_ROOT)  # §6 issue 2

    source = load_patched_eval_model_source(REPO_ROOT)  # §6 issue 1
    source = substitute_questioner_import(source, _QUESTIONER_IMPORT)

    injected_oracle = build_oracle(args.oracle)
    if injected_oracle is not None:
        if _ORACLE_BLOCK not in source:
            raise RuntimeError("Oracle-construction block not found — eval_model.py changed upstream, re-verify ENV.md.")
        source = source.replace(_ORACLE_BLOCK, _INJECTED_ORACLE_BLOCK)

    if os.environ.get("VLM_BACKEND", "vllm") != "fake" or args.oracle in ("upstream", "stub"):
        preflight_vllm()

    print(f"[run_official_eval] oracle={args.oracle} questioner_backend={os.environ.get('VLM_BACKEND', 'vllm')} "
          f"episodes=[{args.start_idx}, {args.end_idx}) type={args.description_type}")

    from llm import CALL_STATS, reset_call_stats

    reset_call_stats()

    old_argv, old_cwd = sys.argv, Path.cwd()
    sys.argv = [
        "eval_model.py", str(args.start_idx), str(args.end_idx),
        "--description-type", args.description_type, "--local", str(args.local),
    ]
    os.chdir(REPO_ROOT)
    try:
        namespace = {
            "__name__": "__main__",
            "__file__": str(REPO_ROOT / "eval_model.py"),
            "_injected_oracle": injected_oracle,
        }
        exec(compile(source, str(REPO_ROOT / "eval_model.py") + " (patched)", "exec"), namespace)  # noqa: S102
    finally:
        sys.argv, _ = old_argv, os.chdir(old_cwd)

    log_data = namespace.get("log_data", {})
    n_attempted = args.end_idx - args.start_idx
    n_logged = len(log_data.get("id", []))
    print()
    print("=" * 70)
    print(f"episodes attempted: {n_attempted}")
    print(f"episodes logged (completed, not discarded): {n_logged}")
    print(f"discarded/errored: {n_attempted - n_logged}  (includes ENV.md §6 bug 3 -- state this denominator)")
    if n_logged:
        print(f"n_successes: {log_data['n_successes']}")
        print(f"n_questions: {log_data['n_questions']}")
        print(f"time_required: {log_data['time_required']}")

    # Did this run compute anything, or replay a warm cache? Without this the two are
    # indistinguishable in the output -- which is how a fully-cached replay got mistaken for a
    # fresh measurement, and (in the same shape) how FakeVLM output got mistaken for real in
    # full_sweep_v1/v2. Always printed, never inferred from wall clock.
    hits, misses = CALL_STATS["hits"], CALL_STATS["misses"]
    total = hits + misses
    print("-" * 70)
    print(f"LLM calls: {total}  (cache hits {hits}, model queries {misses})")
    for model, per in sorted(CALL_STATS["by_model"].items()):
        print(f"  {model}: hits={per['hits']} queries={per['misses']}")
    if total and misses == 0:
        print()
        print("!!! CACHE REPLAY -- the model was never queried. Every answer came from")
        print("!!! artifacts/cache, so this reproduces an earlier run rather than measuring")
        print("!!! anything new. Valid only if the model id AND every prompt are unchanged.")
        print("!!! To force a real run, move the cache aside or change what you are testing.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
