"""Runtime application of the §0.7 upstream fixes. See README.md for what/why.

Nothing here ever writes to the upstream files on disk — `load_patched_eval_model_source`
returns patched *text*, and the caller (`scripts/run_eval.py`) is responsible for exec'ing it.
"""

from pathlib import Path

_BUGGY_LINE = "print(f\"Task is: {info['task_description']}\")"
_FIXED_LINE = "print(f\"Task is: {info['target_description']}\")"


class UpstreamChangedError(RuntimeError):
    """Raised when a patch's expected source text isn't found — the spec explicitly says to
    re-verify §0 facts against upstream rather than silently apply a stale patch (§0: "If the
    upstream repo changes, re-verify before editing this spec").
    """


def load_patched_eval_model_source(repo_root: str | Path) -> str:
    """§0.7 issue 1: `info['task_description']` -> KeyError; should be `target_description`."""
    path = Path(repo_root) / "eval_model.py"
    source = path.read_text()
    if _BUGGY_LINE not in source:
        raise UpstreamChangedError(
            f"Expected buggy line not found in {path} — eval_model.py may have changed upstream. "
            "Re-verify patches/001_task_description_keyerror.diff before proceeding."
        )
    return source.replace(_BUGGY_LINE, _FIXED_LINE)


_QUESTIONER_IMPORT_LINE = "from Questioner import YourQuestioner"


def substitute_questioner_import(source: str, replacement: str = "from agent.questioner import GraphQuestioner as YourQuestioner") -> str:
    """Not a bug fix — an integration-point substitution so the unmodified eval loop (which
    calls `YourQuestioner(info)`) runs our `GraphQuestioner` instead. Kept separate from the
    §0.7 bug patches above since it isn't one; `run_eval.py` applies it on top of them.
    """
    if _QUESTIONER_IMPORT_LINE not in source:
        raise UpstreamChangedError(
            f"Expected import line {_QUESTIONER_IMPORT_LINE!r} not found — eval_model.py may have changed upstream."
        )
    return source.replace(_QUESTIONER_IMPORT_LINE, replacement)


def ensure_episodes_symlink(repo_root: str | Path, run_type: str = "train") -> Path:
    """§0.7 issue 2: eval_model.py looks for QA_eval/episodes_{run_type}.jsonl; the repo ships
    episodes_{run_type}.jsonl at the root. Creates the symlink idempotently.
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
