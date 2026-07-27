#!/usr/bin/env python3
"""Offline data work (no model needed): dumps N target images with their 6 description variants
side by side into a browsable HTML page. Fully runnable now.
"""

import html
import json
import random
from pathlib import Path

DIRECTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DIRECTION_ROOT.parent
EPISODES_PATH = REPO_ROOT / "episodes_train.jsonl"
OUT_PATH = DIRECTION_ROOT / "artifacts" / "description_gallery.html"

DESCRIPTION_TYPES = ["category", "color", "context", "color_feature", "color_context", "color_context_feature"]
N = 20
SAMPLE_SEED = 0


def load_episodes(path: Path = EPISODES_PATH) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def esc(s) -> str:
    return html.escape(str(s), quote=True)


def build_card(ep: dict) -> str:
    rows = "".join(f"<tr><th>{esc(t)}</th><td>{esc(ep['tasks'][t])}</td></tr>" for t in DESCRIPTION_TYPES)
    # OUT_PATH lives at direction_method/artifacts/; images/ is a sibling of direction_method/ at
    # the repo root, so from artifacts/ that's two levels up, then into images/.
    rel_img = "../../" + ep["path"]
    return f'''
<section class="card">
  <div class="card-body">
    <img src="{esc(rel_img)}" loading="lazy" alt="{esc(ep['path'])}">
    <table>{rows}</table>
  </div>
  <div class="meta">id: {esc(ep['id'])} &middot; category: {esc(ep['category'])} &middot; candidates: {len(ep['distractors'])}</div>
</section>'''


def main() -> None:
    episodes = load_episodes()
    sample = random.Random(SAMPLE_SEED).sample(episodes, min(N, len(episodes)))
    cards = "".join(build_card(ep) for ep in sample)

    doc = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>direction_method — description gallery</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#f7f7f8; --card-bg:#fff; --text:#1a1a1a; --muted:#6b6b6b; --border:#e2e2e5; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#15161a; --card-bg:#1f2024; --text:#e8e8ea; --muted:#9a9aa2; --border:#33343a; }}
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--text); }}
main {{ max-width: 900px; margin: 0 auto; padding: 20px 16px 60px; }}
h1 {{ font-size: 16px; }}
.card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 14px; margin-bottom: 16px; }}
.card-body {{ display: flex; gap: 16px; flex-wrap: wrap; }}
img {{ width: 220px; height: 220px; object-fit: cover; border-radius: 8px; flex-shrink: 0; border: 1px solid var(--border); }}
table {{ border-collapse: collapse; font-size: 13px; flex: 1; min-width: 260px; }}
th {{ text-align: left; color: var(--muted); padding: 4px 10px 4px 0; white-space: nowrap; vertical-align: top; font-weight: 600; }}
td {{ padding: 4px 0; }}
.meta {{ font-size: 11px; color: var(--muted); margin-top: 8px; }}
</style>
</head>
<body>
<main>
<h1>{len(sample)} target images &times; 6 description variants (seed={SAMPLE_SEED})</h1>
{cards}
</main>
</body>
</html>
'''
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(doc)
    print(f"Wrote {OUT_PATH} ({len(sample)} episodes)")


if __name__ == "__main__":
    main()
