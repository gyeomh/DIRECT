# DIRECT — CoIN Challenge questioner

**DIRECT** — **DI**rectional **RE**asoning via grounding images with **C**heck**lisT**s — is a
questioner for the [CoIN Challenge](https://github.com/e-zorzi/coin_challenge): a "20 questions"
game where an oracle holds one fixed target image, you are shown candidate images one at a time,
and you must decide which candidate is the same object instance as the target — asking the oracle
as few clarifying questions as possible.

This repo is the upstream challenge harness, unmodified, plus `direction_method/` (DIRECT's
implementation), a solution that works by **spatial context**: parse the description into a target
phrase plus a checklist of claims grouped by spatial relation, ground the target in each candidate
image, ask the oracle about the directions around it, and check every answer against the candidate
before accepting it.

> Upstream's own README is preserved verbatim as [`UPSTREAM_README.md`](UPSTREAM_README.md). It is
> renamed only to avoid a `README.md` / `README.MD` collision on case-insensitive filesystems; the
> contents are byte-identical to upstream.

---

## How it works

Four modules, all backed by the same VLM, wired together by `DirectionMethodQuestioner`:

| module | input | output |
|---|---|---|
| `context_parser` | the description (text only) | target category, target phrase, initial checklist |
| `zone_gen` | candidate image + target category | bounding box, then which surrounding directions are worth asking about |
| `self_check` | candidate image + (region, claim) | evidence + a yes/no verdict |
| `checklist_update` | checklist + this round's (relation, answer) pairs | grown checklist (no LLM call) |

`color_family.py` supports `self_check`: when it returns a contradiction, `self_check` reads colors
accurately but does not reliably apply "same color family" as a single rule regardless of how the
rule is worded (measured — three separate prompt rewrites produced byte-identical verdicts). The
claimed and perceived color terms are extracted from the assertion/evidence text and compared
against a hue+lightness family table in code instead; this can only turn a `"no"` into a `"yes"`,
never the reverse, so it cannot introduce a new false rejection.

The loop, per candidate:

1. **Re-verify** every claim already in the checklist against this candidate. One `self_check` per
   claim; the first failure concludes *mismatch* immediately.
2. **Find zones.** Ground the target, draw its box, ask which directions around it are informative.
   Directions already covered by the checklist are dropped — so each direction is asked **at most
   once per episode**.
3. **Ask and verify.** One oracle question per remaining direction, each answer checked by
   `self_check` against the candidate. First failure concludes *mismatch*.
4. **Grow the checklist** with every answer received — including the one that just failed, since
   the oracle only ever describes the true target, making a failing answer the most discriminative
   fact available for later candidates.

The checklist persists for the whole episode and only grows, so later candidates are rejected by
cheap image checks instead of new questions.

Full design rationale, including what was tried and rejected: [`direction_method/SPEC.md`](direction_method/SPEC.md).
Harness facts the design is built around: [`direction_method/ENV.md`](direction_method/ENV.md).

---

## Requirements

- **A GPU** able to hold the served model (~71 GB for the default; a single 80 GB card is enough).
- **Two separate Python environments.** The model server and the client code have conflicting
  dependency trees — do not merge them.
- **Python 3.11** for the client env (3.14 will fail on upstream's pinned `transformers`).

---

## Setup

### 1. Clone and fetch the images

```bash
git clone <this-repo> coin_challenge
cd coin_challenge
mkdir images && hf download --repo-type dataset e-zorzi/images_coin_challenge --local-dir images
```

`images/` is gitignored — it is a real dataset download, not part of this repo. If the dataset is
gated for your account, run `huggingface-cli login` first. Confirm `images/` is non-empty before
continuing; every episode resolves its paths relative to the repo root.

### 2. Client environment

```bash
python3.11 -m venv coin_env && source coin_env/bin/activate
pip install retrying flask attrs gymnasium colorama accelerate transformers==4.43.1 Pillow \
    opencv-python dotenv qwen-vl-utils huggingface_hub google-genai openai numpy pytest
```

That is upstream's own `scripts/install.sh` list plus `pytest`, which the test suite needs.

Verify without a GPU — the whole pipeline runs against a canned offline backend:

```bash
python -m pytest direction_method/tests -q     # 298 passed
```

### 3. Serving environment

```bash
python -m venv vllm_env && source vllm_env/bin/activate
pip install 'vllm>=0.19.0' ninja
```

Both pins matter:

- **`vllm>=0.19.0`** — older versions do not know the default model's architecture class. Pin the
  exact version if `pip` pulls a `torch` built for a CUDA your driver does not have; check
  `python -c "import torch; print(torch.cuda.is_available())"` before going further.
- **`ninja`** — the default model's attention kernel is JIT-compiled by flashinfer on the *first
  real prefill*. Without `ninja` the engine dies there **while the API process stays up and keeps
  answering `/v1/models` with 200**, so a run started against it looks healthy and silently
  produces nothing.

### 4. Start the model server

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --port 8002 --max-model-len 8192 --limit-mm-per-prompt.video 0 \
    --enable-prefix-caching --gpu-memory-utilization 0.85
```

Set `CUDA_VISIBLE_DEVICES` to whichever GPU you intend to use. Startup takes a few minutes; wait
for `Application startup complete`.

**Prefix caching is not optional for performance.** Every `self_check` call for one candidate shares
the same image, and the image is placed at the very front of the prompt, so its prefill is paid once
per candidate instead of once per claim.

---

## Running

### The official evaluation path

This drives upstream's own `eval_model.py` — the same code path the real evaluation uses — rather
than reimplementing the candidate loop:

```bash
source coin_env/bin/activate
VLM_PORT=8002 python direction_method/scripts/run_official_eval.py 0 167 \
    --oracle stub --description-type all
```

- `0 167` — episode range (167 is the whole training set).
- `--description-type` — one of `category`, `color`, `context`, `color_feature`, `color_context`,
  `color_context_feature`, or `all` to loop over all six (= 1002 runs).
- `--oracle` — `stub` uses a local stand-in on the same server (no API cost); `upstream` uses
  upstream's own construction (Gemini, or a local VLM via `ORACLE_MODEL_ID` + `--local 1`);
  `mock` returns one fixed string and is for plumbing tests only.

Upstream files are **never edited**. The ENV-level fixes are applied to `eval_model.py`'s source
*in memory* before it is executed — see [`direction_method/patches/README.md`](direction_method/patches/README.md).

Before running anything, the script refuses to start unless the server is actually usable: it
checks that the served model is the one the questioner will request, and issues a real completion
(listing models is not enough — see the `ninja` note above).

Results land in `results/` as upstream's gzip logs. Summarize them:

```bash
python direction_method/scripts/summarize_official.py \
    --oracle-model-id Qwen/Qwen3.6-35B-A3B-FP8
```

### The parallel sweep (development)

Faster, resumable, and writes rich per-episode traces, but reimplements the loop:

```bash
SWEEP_LOG_ROOT=/path/to/logs VLM_PORT=8002 python direction_method/scripts/run_full_sweep.py
```

**Do not raise `SWEEP_WORKERS` above its default of 3** for the default model. At 6 and at 12 the
server wedges partway through a sweep — requests time out and the GPU stays pinned until the
*server* process is `kill -9`'d, so it is a stuck server-side generation, not client queueing. The
root cause is not isolated.

Browse a sweep's traces in a local viewer:

```bash
VIEWER_LOG_ROOT=/path/to/logs python direction_method/scripts/serve_viewer.py
```

---

## Caching: on for development, off for measurement

Every call runs at `temperature=0.0`, so `LLMClient` can cache responses on disk keyed by
`sha256(model | prompt | image_hash)`. This makes iterating on one module fast.

**It is off by default in `run_official_eval.py`, and should stay off for any number you report.**
vLLM is not bit-deterministic even at temperature 0 — batching, chunked prefill and kernel
scheduling move results between runs. Caching freezes one draw from that distribution, so a cached
sweep cannot show run-to-run variance, and a change that alters no prompt text (post-processing,
for instance) yields byte-identical "results" no matter what it actually did.

Every run prints its call accounting, so a replay can never be mistaken for a measurement:

```
LLM calls: 30  (cache hits 30, model queries 0)

!!! CACHE REPLAY -- the model was never queried. ...
```

Relevant knobs, all env vars because `eval_model.py` constructs the questioner as
`YourQuestioner(info)` with no way to thread arguments through:

| variable | meaning |
|---|---|
| `VLM_PORT` | server port (default 8000) |
| `VLM_MODEL_ID` | override the model the questioner requests |
| `VLM_USE_CACHE=1` | re-enable the disk cache (off in the eval path) |
| `VLM_CACHE_DIR` | where cache entries live |
| `VLM_BACKEND=fake` | offline canned backend; no server, no GPU |

`VLM_BACKEND=fake` writes into a separate `_fake/` cache namespace so placeholder output can never
be replayed as a real result.

---

## Scoring

Upstream states the objective in words — *"maximize the number of correct conclusions while asking
as few questions as possible"* — but defines no formula. Two are reported:

- **`10 - number_of_questions`** per fully-correct episode, `0` for a failure. An episode counts
  only if **every** candidate in it was judged correctly.
- **`env.py`'s own reward**: `+10` per correct conclusion, `-10` per wrong one, `-1` per question.
  This is the only numeric scoring defined anywhere in the repo.

Under either, questions are a direct deduction, and they dominate: at ~3.6 questions per episode a
won episode scores about 6.4 out of 10.

**State the denominator.** On the training set the harness sometimes fails to advance to the next
candidate after a correct conclusion and the episode is discarded (a known upstream bug, fixed for
the held-out set). That is ~9-10 episodes per description type. The summarizer reports
`_discarded_episodes` and `_accuracy_over_logged` next to the headline numbers so this is never
silently absorbed.

---

## Repo layout

```
env.py, eval_model.py, Questioner.py, utils.py, Oracle.py, episodes_train.jsonl
                              upstream harness — byte-identical, never edited
UPSTREAM_README.md            upstream's README, verbatim

direction_method/
  SPEC.md                     design, experiments, and what was rejected
  ENV.md                      fixed facts about the harness
  HANDOFF.md                  bringing this up on a fresh GPU machine
  questioner.py               the QuestionerInterface implementation
  context_parser.py  zone_gen.py  self_check.py  checklist_update.py
  color_family.py              color-family reconciliation self_check calls into
  templates.py                shared region/question wording, one table for both paths
  llm.py                      vllm client, disk cache, call accounting
  oracle_stub.py              local oracle stand-in (sees only the target image)
  patches/                    upstream bug fixes, applied in memory at runtime
  scripts/                    eval drivers, per-module experiments, log viewer
  tests/                      298 tests, all runnable with no GPU
```

---

## Troubleshooting

**`NotImplementedError: Insert here your Questioner class`** — almost never means what it says.
`eval_model.py` wraps questioner construction in a bare `except:` and re-raises this fixed message,
swallowing the real error (usually the server being down or on another port). `run_official_eval.py`
preflights the server to avoid exactly this; if you hit it elsewhere, check the server first.

**A run finishes suspiciously fast and the GPU never moves** — check the printed call accounting.
`cache hits N, model queries 0` means it replayed the cache instead of computing anything.

**The server answers `/v1/models` but every request fails** — the API process outlives a dead
engine. Check the server log for a `flashinfer` JIT failure (`ninja`), then restart it.

**`torch.cuda.is_available()` is `False` after installing vllm** — `pip` pulled a `torch` built for
a newer CUDA than the driver. Pin the vllm version so it resolves a matching `torch`.

---

## Credits

Challenge, harness, and dataset by [e-zorzi](https://github.com/e-zorzi/coin_challenge).
`direction_method/` is this repo's own contribution.
