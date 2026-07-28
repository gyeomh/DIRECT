# HANDOFF.md — running direction_method on a GPU server

You are Claude Code, running on a machine with GPU access, helping someone get this project's
`direction_method` running for real (against a live vllm server) instead of the offline
`FakeVLM` backend it was built and unit-tested against on a machine with no GPU. Read this file
top to bottom before doing anything. Then read `SPEC.md` (design) and `ENV.md` (harness facts) —
both ship inside this same `direction_method/` folder.

**What you were handed**: only the `direction_method/` folder, zipped. It depends on the upstream
CoIN Challenge repo, which is NOT included and which you need to set up first. Everything below
assumes you start from nothing but this folder.

---

## 0. What this project is, briefly

`direction_method` is one candidate solution to the CoIN Challenge (a "20 questions"-style game:
an oracle describes a target image; a questioner sees a sequence of candidate images and must
decide, asking the oracle clarifying questions when unsure, which candidate matches the target).
It works by parsing the target description into a target phrase + a checklist of atomic
assertions grouped by spatial relation (`context_parser`), grounding the target object in each
candidate image and picking which surrounding directions are worth asking about
(`zone_gen`), verifying every assertion/answer against the candidate image via a
contradiction-framed yes/no check (`self_check`), and folding newly-learned facts back into the
checklist (`checklist_update`). `questioner.py` wires all four together into the harness's
`QuestionerInterface`.

Everything so far was built and unit-tested against `FakeVLM` (`llm.py`'s offline stand-in
backend, canned schema-conforming responses, no network calls) because development happened on a
machine with no GPU access. **Nothing has been verified against a real model yet.** That's your
job. `SPEC.md` §11 (Open items) lists exactly what's still unconfirmed; §13's own note explains
why the FakeVLM dry run over all 167 training episodes shows every episode resolving in exactly
one candidate (expected, not a bug — FakeVLM's `self_check` can never actually produce a
mismatch).

---

## 1. Get the upstream repo (not included in this zip)

```bash
git clone https://github.com/e-zorzi/coin_challenge.git
cd coin_challenge
```

Now unzip this handoff package so `direction_method/` sits **as a sibling of `env.py`** — i.e.
`coin_challenge/direction_method/...`, not nested one level deeper. Confirm:

```bash
ls              # should show: Oracle.py Questioner.py env.py eval_model.py utils.py
                 # episodes_train.jsonl README.MD scripts/ direction_method/ ...
```

Everything direction_method's code imports from the repo root (`env.py`, `Questioner.py`,
`Oracle.py`, `utils.py`) and `episodes_train.jsonl` come from this clone, unmodified — the whole
project's philosophy (see `ENV.md` and `direction_method/patches/README.md`) is to never edit
upstream files, only patch them at runtime in memory. You should not need to touch anything
outside `direction_method/`.

### Images

Upstream's own `README.MD` (now sitting at the repo root) explains this, but concretely:

```bash
mkdir images
hf download --repo-type dataset e-zorzi/images_coin_challenge --local-dir images
```

Requires the `huggingface_hub` package (`pip install huggingface_hub`, or it comes with the env
setup in the next section) and the `hf` CLI it provides. If the dataset is gated for you, run
`huggingface-cli login` first. This is a real download (image files) — expect it to take a while
and check the resulting `images/` directory looks populated (not empty) before moving on.

---

## 2. Python environments

Upstream's `scripts/install.sh` and `scripts/install_vllm.sh` (at the repo root) create two plain
`venv` environments; use those, or conda equivalents — whatever this machine already prefers.
**Keep the VLM-serving environment separate from the one that imports `direction_method`** (the
model server and the client code have very different, easily-conflicting dependency trees).

**Base env** (anything importing `direction_method` or running its scripts/tests):

```bash
python -m venv coin_env && source coin_env/bin/activate
pip install retrying flask attrs gymnasium colorama accelerate transformers==4.43.1 Pillow \
    opencv-python dotenv qwen-vl-utils huggingface_hub google-genai openai numpy pytest pyflakes
```

(That's upstream's own `install.sh` list, plus `pytest`/`pyflakes` which `direction_method`'s test
suite and lint checks need but upstream doesn't.)

**vllm env** (only for serving the model):

```bash
python -m venv vllm_env && source vllm_env/bin/activate
pip install vllm
```

**CUDA/driver compatibility — check this fresh, don't assume.** On the machine this was built on,
a plain `pip install vllm` pulled a version requiring CUDA 13, but the GPU driver there only
supported CUDA 12.8. If you hit something like that: `vllm==0.15.0` is confirmed to work with
`Qwen/Qwen3-VL-30B-A3B-Instruct` (it pulls `torch==2.9.1`, whose default PyPI build is `+cu128`,
and it already ships the `Qwen3VLMoeForConditionalGeneration` architecture). Check `nvidia-smi`
for the driver's max supported CUDA version before deciding whether you need this pin at all —
your GPU/driver may be newer and need no pin.

---

## 3. GPU usage

Check `nvidia-smi` first to see what's actually free — don't assume GPU 0. Pin every GPU-bound
process explicitly with `CUDA_VISIBLE_DEVICES=<N>` and re-check `nvidia-smi` after starting each
one to confirm it landed where you expected and nothing else is competing for that device.

One vllm server can serve **both** roles at once (the questioner's own VLM calls, and the local
oracle stand-in that answers questions by looking at the target image) — that's fine for
correctness testing. `SPEC.md` §8 notes that real *latency* numbers need the oracle on a separate
GPU/server (so its inference doesn't compete with the modules being timed) — only matters once
you're past correctness and into performance measurement.

---

## 4. Serve the model

```bash
source vllm_env/bin/activate
CUDA_VISIBLE_DEVICES=<N> vllm serve Qwen/Qwen3-VL-30B-A3B-Instruct \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --limit-mm-per-prompt.video 0 \
    --port 8000
```

(`--max-model-len 8192` and `--limit-mm-per-prompt.video 0` are `SPEC.md` §8's own serving notes;
adjust `--gpu-memory-utilization` down if you're sharing the GPU with anything else.) Wait for it
to report ready before running anything against it — upstream's own launch script sleeps 120s
after starting the server; watch the server's own log output rather than trusting a fixed sleep.

`direction_method/llm.py`'s `LLMClient` defaults to port 8000 and expects an OpenAI-compatible
`/v1` endpoint at `http://localhost:{port}`, matching `utils.py`'s `ClientBasedLLM` — if you serve
on a different port, pass `port=<N>` when constructing `LLMClient` in whichever script you're
running.

---

## 5. Running things

Everything below assumes the base env (`coin_env`) is active and your CWD is
`coin_challenge/direction_method/` (the scripts resolve the repo root from `__file__`, so CWD
mostly doesn't matter, but this is the convention used during development).

**Do not set `VLM_BACKEND=fake`.** `LLMClient` defaults to the real `vllm` backend already —
`fake` was only ever used for offline development without a GPU. Just run the scripts directly
once the server from step 4 is up.

### 5.1 — Sanity check first (no GPU needed at all)

```bash
python3 -m pytest tests/ -q      # should show "140 passed" against FakeVLM
python3 -m pyflakes .             # should be silent
```

This confirms the code survived the transfer intact before you spend any GPU time. If this fails,
something broke in transit or in environment setup — fix that before going further.

### 5.2 — The real, GPU-dependent verification runs

These are `SPEC.md`'s own "verify tomorrow" deliverables — this *is* "tomorrow." Run them in this
order; each writes to `direction_method/artifacts/` (gitignored, safe to regenerate).

1. **`python3 scripts/verify_zone_gen.py`** — runs `zone_gen` on 20-30 real training images.
   Writes `artifacts/zone_gen_gallery.html` (open it — this is the important one to actually
   *look at*) plus `artifacts/zone_gen_verify/boxed/*.png`. Report back:
   - Is the box on the correct object in each case?
   - Is the "one box per run" rule respected (one box over e.g. a whole cabinet row, not one box
     per door)? This is the item SPEC.md calls out as mattering most.
   - Does `scene` correctly name which edges the target touches?
   - **The bbox coordinate convention is currently unverified** (`SPEC.md` §5-1,
     `zone_gen._bbox_to_pixels`/`_EDGE_FRAME`) — it assumes Qwen3-VL returns coordinates relative
     to a `[0, 1000]` grid, but `smart_resize` can shift that. If boxes in the gallery look
     obviously wrong (wrong scale, off-frame, tiny), this assumption is the first thing to check
     and fix.

2. **`python3 scripts/run_self_check_experiment.py`** — reports the false-`"no"` rate (should be
   low; ground truth is always `"yes"` here) overall and per question type, dumps every failure to
   `artifacts/self_check_failures.json`. Read the dump by hand; don't auto-classify it.

3. **`python3 scripts/run_context_parser_examples.py`** — prints `target_category` /
   `target_phrase` / `checklist` for a handful of descriptions, including `SPEC.md` §10's own
   worked examples. Eyeball the output against those examples.

4. **`python3 scripts/run_checklist_update_examples.py`** — prints before/after checklists for a
   handful of hand-built cases (compound-answer splitting, empty regions, duplicate-skipping,
   hedge-preservation). Eyeball against the rule descriptions in `SPEC.md` §12.

5. **Confirm guided decoding actually works.** `llm.py`'s `VLLMBackend.generate` sends
   `extra_body={"guided_json": schema}` — this is flagged in the code as unconfirmed against a
   real server; some vllm versions instead want
   `response_format={"type": "json_schema", "json_schema": {...}}`. If any of the scripts above
   produce malformed JSON, schema-violating output, or `PARSE_ERROR` verdicts, this is the first
   thing to check — see the TODO comment right above that line in `llm.py`.

6. **Optional: confirm prefix caching engages.** `llm.py`'s `check_prefix_caching()` is written
   but never invoked (needed a live server). Its own docstring has a 3-line usage example.

7. **Optional, heavier: the full episode loop against the real backend.**
   `scripts/run_full_loop_dry_run.py` was only ever run against `FakeVLM` (167/167 episodes, zero
   crashes — see `SPEC.md` §13). Running it against a real server means real GPU time across many
   VLM calls per episode; **before pointing it at all 167 episodes, edit `N_EPISODES` near the top
   of the script down to something small (5-10) for an initial timing/sanity check**, then scale
   up once you know how long each episode actually takes.

---

## 6. Reporting back

For each script above, send back:
- The printed console output.
- The generated artifacts (`artifacts/zone_gen_gallery.html` + its `boxed/` images,
  `artifacts/self_check_failures.json`) — these are gitignored, so they won't show up in a diff.
- Any code changes you had to make to get things working (e.g. `guided_json` →
  `response_format`, a corrected `_bbox_to_pixels` conversion, a different `max_pixels` value).
  Keep a copy of this folder as it was unzipped if you want to produce a clean diff of your
  changes — there's no git repo bundled with this handoff.

`SPEC.md` §11 (Open items) is the authoritative list of what's still undecided; update it (or at
least tell whoever reads your results which items got resolved and how) rather than leaving your
findings only in scrollback.
