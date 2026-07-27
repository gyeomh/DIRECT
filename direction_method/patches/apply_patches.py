"""Runtime application of the ENV.md §6 upstream fixes (+ the MockOracle bug found in step 1).
See README.md for what/why. Nothing here writes to the upstream files on disk.
"""

from pathlib import Path

_BUGGY_LINE = "print(f\"Task is: {info['task_description']}\")"
_FIXED_LINE = "print(f\"Task is: {info['target_description']}\")"

_QUESTIONER_IMPORT_LINE = "from Questioner import YourQuestioner"


class UpstreamChangedError(RuntimeError):
    """Raised when a patch's expected source text isn't found — re-verify ENV.md against
    upstream rather than silently apply a stale patch (ENV.md's own header: "If upstream
    changes, re-verify this document first").
    """


def load_patched_eval_model_source(repo_root: str | Path) -> str:
    """ENV.md §6 issue 1: info['task_description'] -> KeyError; should be target_description."""
    path = Path(repo_root) / "eval_model.py"
    source = path.read_text()
    if _BUGGY_LINE not in source:
        raise UpstreamChangedError(
            f"Expected buggy line not found in {path} — eval_model.py may have changed upstream."
        )
    return source.replace(_BUGGY_LINE, _FIXED_LINE)


def substitute_questioner_import(source: str, replacement: str) -> str:
    """Not a bug fix — rewrites the one import line (in memory only) so eval_model.py's
    `YourQuestioner(info)` instantiates whichever class `replacement` names instead.
    """
    if _QUESTIONER_IMPORT_LINE not in source:
        raise UpstreamChangedError(
            f"Expected import line {_QUESTIONER_IMPORT_LINE!r} not found — eval_model.py may have changed upstream."
        )
    return source.replace(_QUESTIONER_IMPORT_LINE, replacement)


def ensure_episodes_symlink(repo_root: str | Path, run_type: str = "train") -> Path:
    """ENV.md §6 issue 2: eval_model.py looks for QA_eval/episodes_{run_type}.jsonl; the repo
    ships episodes_{run_type}.jsonl at the root. Creates the symlink idempotently.
    """
    repo_root = Path(repo_root)
    source = repo_root / f"episodes_{run_type}.jsonl"
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist — nothing to symlink to.")

    qa_eval_dir = repo_root / "QA_eval"
    qa_eval_dir.mkdir(exist_ok=True)
    link = qa_eval_dir / f"episodes_{run_type}.jsonl"
    if link.is_symlink() or link.exists():
        return link
    link.symlink_to(source.resolve())
    return link


class WorkingMockOracle:
    """Drop-in replacement for env.py's MockOracle, which is missing `self` on `ask` and cannot
    be called as shipped (see README.md issue 4). Same fixed-answer behavior, correct signature.
    Not upstream code — lives entirely in our patches/, never written back to env.py.
    """

    def ask(self, *, prompt: str = "", images=None):
        return "Yes that is true [Mock answer]"


_TYPO = "correnctly"
_FIXED_TYPO = "correctly"


def fix_answer_prompt_typo() -> None:
    """Monkeypatches env.py's module-level ANSWER_PROMPT constant *in memory* — not a file edit.
    env._get_observation() looks up ANSWER_PROMPT as a global at call time, so reassigning the
    module attribute after import is sufficient; no exec-the-patched-source trick needed here
    (that's only required for eval_model.py's __main__-time bugs, where the buggy line lives
    inside a function body executed once at script-run time, not a plain constant).

    Keep upstream's ANSWER_PROMPT exactly as-is except the one typo: "correnctly" -> "correctly".
    Nothing else about the prompt changes.

    IMPORTANT: the official harness does not apply this patch and therefore keeps the typo. Any
    local run that calls this is not byte-for-byte identical to the official oracle prompt.
    """
    import env

    if _TYPO not in env.ANSWER_PROMPT:
        raise UpstreamChangedError(
            f"Expected typo {_TYPO!r} not found in env.ANSWER_PROMPT — re-verify ENV.md/env.py."
        )
    env.ANSWER_PROMPT = env.ANSWER_PROMPT.replace(_TYPO, _FIXED_TYPO)
