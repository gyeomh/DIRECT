#!/usr/bin/env python3
"""Thin wrapper over upstream eval_model.py (spec §2, §10 step 2).

STUB: patch application and process wiring are in place; the discard-accounting / accuracy
summary below is intentionally minimal until there's a real questioner to generate numbers worth
tabulating (extract.py / adjudicate.py prompts are still TODO). Fill in the TODOs once those land.

Usage mirrors eval_model.py plus our own flags:
    python scripts/run_eval.py <start_idx> <end_idx> [--description-type ...] [--local 0|1]
"""

import sys
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _PACKAGE_ROOT.parent
for p in (_PACKAGE_ROOT, _REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from patches.apply_patches import (  # noqa: E402
    ensure_episodes_symlink,
    load_patched_eval_model_source,
    substitute_questioner_import,
)


def run(argv: list[str]) -> dict:
    """Applies the §0.7 patches + questioner substitution, then execs eval_model.py's source
    in a fresh namespace with `sys.argv` set to `argv` (eval_model.py parses argparse at import
    time, so this has to happen before exec, not after).

    Returns the exec namespace's `log_data` dict — TODO(next): once GraphQuestioner produces real
    conclusions, compute per-description-type accuracy from it here, report
    `len(log_data['id'])` against `end_idx - start_idx` as the discard count (spec §0.7 issue 3,
    `min_eval_episodes` check from config.yaml), and refuse to report accuracy below
    `config.yaml: eval.min_eval_episodes` post-discard.
    """
    ensure_episodes_symlink(_REPO_ROOT)
    source = load_patched_eval_model_source(_REPO_ROOT)
    source = substitute_questioner_import(source)

    old_argv = sys.argv
    sys.argv = ["eval_model.py", *argv]
    try:
        namespace = {"__name__": "__main__", "__file__": str(_REPO_ROOT / "eval_model.py")}
        code = compile(source, str(_REPO_ROOT / "eval_model.py") + " (patched)", "exec")
        exec(code, namespace)  # noqa: S102 — intentional: this *is* the wrapper's job
    finally:
        sys.argv = old_argv

    return namespace.get("log_data", {})


if __name__ == "__main__":
    # eval_model.py owns its own argparse; we just forward argv through unchanged.
    run(sys.argv[1:])
