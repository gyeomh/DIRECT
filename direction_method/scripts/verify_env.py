"""ENV.md step 1: verify every fact against the installed repo, then run eval_model.py itself
(patched, per patches/) end to end with MockOracle + a throwaway questioner. No method code.
"""

import json
import sys
from collections import Counter
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


def verify_section_7(episodes_path: Path) -> bool:
    """ENV.md §7's exact check, plus the position-distribution cross-tabulation."""
    print("=" * 70)
    print("§7 data verification: does match == target, exactly one per episode?")
    print("=" * 70)

    episodes = []
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    print(f"total episodes: {len(episodes)}")

    n_not_exactly_one = 0
    n_path_mismatch = 0
    positions = []
    for ep in episodes:
        m = [d for d in ep["distractors"] if d["match"]]
        if len(m) != 1:
            n_not_exactly_one += 1
            print(f"  MISMATCH (not exactly 1 match): {ep['id']} has {len(m)}")
        if not all(d["path"] == ep["path"] for d in m):
            n_path_mismatch += 1
            print(f"  MISMATCH (match path != target path): {ep['id']}")
        for i, d in enumerate(ep["distractors"]):
            if d["match"]:
                positions.append(i)

    n_candidates_dist = Counter(len(ep["distractors"]) for ep in episodes)
    position_dist = Counter(positions)

    print(f"episodes with != 1 match: {n_not_exactly_one}")
    print(f"episodes where match path != target path: {n_path_mismatch}")
    print()
    print("[AWARENESS ONLY — no component may consume this] match position distribution:")
    print(" ", dict(sorted(position_dist.items())))
    print("candidates-per-episode distribution (cross-check):")
    print(" ", dict(sorted(n_candidates_dist.items())))
    always_last = all(position_dist.get(n - 1, 0) == count for n, count in n_candidates_dist.items())
    print(f"match is always the last candidate in every episode: {always_last}")

    ok = n_not_exactly_one == 0 and n_path_mismatch == 0
    print("RESULT:", "PASS — matches ENV.md §7's assumption exactly" if ok else "FAIL — see mismatches above")
    return ok


def verify_mock_oracle_bug():
    print()
    print("=" * 70)
    print("Upstream MockOracle.ask() bug (not in ENV.md, found during step 1)")
    print("=" * 70)
    from env import MockOracle

    try:
        MockOracle().ask(prompt="hi", images=[])
        print("RESULT: MockOracle.ask() worked?! (expected TypeError — re-verify ENV.md)")
    except TypeError as e:
        print(f"Confirmed: {e}")
        print("Using patches.apply_patches.WorkingMockOracle instead (see patches/README.md #4).")


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

_TEST_ORACLE_BLOCK = "    oracle_client = _test_oracle  # substituted for verification only, see verify_env.py"


def run_end_to_end(start_idx: int, end_idx: int, description_type: str = "category"):
    print()
    print("=" * 70)
    print(f"End-to-end run: eval_model.py (patched) + MockOracle + TrivialQuestioner, episodes [{start_idx}, {end_idx})")
    print("=" * 70)

    ensure_episodes_symlink(REPO_ROOT)
    source = load_patched_eval_model_source(REPO_ROOT)
    source = substitute_questioner_import(
        source, "from patches.trivial_questioner import TrivialQuestioner as YourQuestioner",
    )
    if _ORACLE_BLOCK not in source:
        raise RuntimeError("Oracle-construction block text not found — eval_model.py may have changed upstream.")
    source = source.replace(_ORACLE_BLOCK, _TEST_ORACLE_BLOCK)

    old_argv = sys.argv
    sys.argv = ["eval_model.py", str(start_idx), str(end_idx), "--description-type", description_type]
    old_cwd = Path.cwd()
    import os
    os.chdir(REPO_ROOT)
    try:
        namespace = {
            "__name__": "__main__",
            "__file__": str(REPO_ROOT / "eval_model.py"),
            "_test_oracle": WorkingMockOracle(),
        }
        code = compile(source, str(REPO_ROOT / "eval_model.py") + " (patched, test oracle)", "exec")
        exec(code, namespace)  # noqa: S102 — intentional, this is the verification harness's job
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    log_data = namespace.get("log_data", {})
    n_attempted = end_idx - start_idx
    n_logged = len(log_data.get("id", []))
    print()
    print(f"episodes attempted: {n_attempted}")
    print(f"episodes logged (completed without error/discard): {n_logged}")
    print(f"discarded/errored: {n_attempted - n_logged} (includes ENV.md §6 bug 3 occurrences, if any)")
    if n_logged:
        print(f"n_successes per logged episode: {log_data['n_successes']}")
        print(f"n_questions per logged episode: {log_data['n_questions']}")
    return namespace


if __name__ == "__main__":
    ok7 = verify_section_7(REPO_ROOT / "episodes_train.jsonl")
    verify_mock_oracle_bug()
    run_end_to_end(0, 5, description_type="category")
    print()
    print("=" * 70)
    print("DONE. §7 data check:", "PASS" if ok7 else "FAIL — see above")
    print("=" * 70)
