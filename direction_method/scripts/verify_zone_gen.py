"""zone_gen verification, per the "Verify tomorrow, 20-30 training images" section of the
zone_gen build instructions. Built now, meant to run unattended tomorrow with a live vllm server:
`python scripts/verify_zone_gen.py`.

No checklist exists yet (context_parser/checklist_update aren't built), so `resolve_relations` is
called with `existing_parent_keys=set()` -- every episode here is effectively "first candidate."

What this script CAN measure automatically:
  - box count distribution (n_boxes > 1 is an ambiguity signal, SPEC.md §5)
  - region key / region count distribution returned by zone_gen's 5-2 call
  - how often a directional key whose edge the box touches still comes back from the VLM
    (the edge_touch_log leak rate) and how often the geometry fallback had to fire

What this script CANNOT measure automatically -- these need a human reading the dumped gallery:
  - is the box on the correct object
  - is the "one box per run" rule respected (one box over a cabinet row, not a single door)
  - does the "scene" text actually name the correct touched edges

Does not auto-classify any of the human-judgment items. Reading the gallery is a human step.
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

DIRECTION_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = DIRECTION_ROOT.parent
for p in (DIRECTION_ROOT, REPO_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
from PIL import Image

from llm import LLMClient
from zone_gen import locate, resolve_relations, zones

MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
N_EPISODES = 25
SAMPLE_SEED = 0

OUT_DIR = DIRECTION_ROOT / "artifacts"
BOXED_DIR = OUT_DIR / "zone_gen_verify" / "boxed"
GALLERY_PATH = OUT_DIR / "zone_gen_gallery.html"


def load_episodes(path: Path) -> list[dict]:
    episodes = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                episodes.append(json.loads(line))
    return episodes


def main() -> None:
    episodes = load_episodes(REPO_ROOT / "episodes_train.jsonl")
    sample = random.Random(SAMPLE_SEED).sample(episodes, min(N_EPISODES, len(episodes)))

    llm_client = LLMClient(MODEL_ID, cache_dir=DIRECTION_ROOT / "artifacts" / "cache")

    n_boxes_dist = Counter()
    region_count_dist = Counter()  # len(raw_regions) per episode, pre-dedup
    region_key_dist = Counter()  # every raw key returned, across all episodes
    n_fallback = 0
    n_edge_leak = 0  # episodes where edge_touch_log is non-empty
    rows = []  # for the gallery

    BOXED_DIR.mkdir(parents=True, exist_ok=True)

    for ep in sample:
        target_category = ep["category"]  # SPEC §5-1: category noun only, not the full description
        image = np.array(Image.open(REPO_ROOT / ep["path"]).convert("RGB"))

        loc = locate(llm_client, image, target_category)
        zr = zones(llm_client, loc.boxed_image, target_category)
        resolved = resolve_relations(zr, loc.bbox_2d, existing_parent_keys=set())

        n_boxes_dist[loc.n_boxes] += 1
        region_count_dist[len(zr.raw_regions)] += 1
        for r in zr.raw_regions:
            region_key_dist[r["key"]] += 1
        if resolved.used_fallback:
            n_fallback += 1
        if resolved.edge_touch_log:
            n_edge_leak += 1

        boxed_path = BOXED_DIR / f"{ep['id']}.png"
        Image.fromarray(loc.boxed_image).save(boxed_path)

        rows.append({
            "id": ep["id"],
            "category": target_category,
            "boxed_image_path": boxed_path,
            "n_boxes": loc.n_boxes,
            "bbox_2d": loc.bbox_2d,
            "scene": zr.scene,
            "raw_regions": zr.raw_regions,
            "final_relations": resolved.relations,
            "used_fallback": resolved.used_fallback,
            "edge_touch_log": resolved.edge_touch_log,
        })

    n = len(sample)
    print("=" * 70)
    print(f"zone_gen verification: {n} episodes, backend={llm_client.backend_name}, model={MODEL_ID}")
    print("=" * 70)
    print("\nbox count distribution (n_boxes=1 is expected; >1 is an ambiguity signal):")
    print(f"  {dict(sorted(n_boxes_dist.items()))}")
    print(f"\nregion count distribution (raw, pre-dedup, per episode): {dict(sorted(region_count_dist.items()))}")
    print(f"region key distribution (raw, across all episodes): {dict(region_key_dist.most_common())}")
    print(f"\ngeometry fallback fired: {n_fallback}/{n} episodes")
    print(f"edge-touch leak (VLM returned a direction whose edge the box touches): {n_edge_leak}/{n} episodes")
    print("\nNOT measured automatically -- read the gallery by hand for each of these:")
    print("  - is the box on the correct object")
    print("  - is the one-box-per-run rule respected (not one box per door/segment)")
    print("  - does 'scene' correctly name the touched edges")

    _write_gallery(rows)
    print(f"\nGallery written to {GALLERY_PATH}")


def _write_gallery(rows: list[dict]) -> None:
    import html as htmlmod

    def esc(s) -> str:
        return htmlmod.escape(str(s), quote=True)

    def build_card(row: dict) -> str:
        rel_img = "zone_gen_verify/boxed/" + row["boxed_image_path"].name
        regions_html = "".join(f"<li>{esc(r['key'])} — {esc(r['note'])}</li>" for r in row["raw_regions"])
        final_html = ", ".join(esc(k) for k in row["final_relations"]) or "(none)"
        edge_leak_html = ", ".join(esc(k) for k in row["edge_touch_log"]) or "(none)"
        return f'''
<section class="card">
  <div class="card-body">
    <img src="{esc(rel_img)}" loading="lazy" alt="{esc(row['id'])}">
    <div class="info">
      <div><b>category:</b> {esc(row['category'])}</div>
      <div><b>n_boxes:</b> {row['n_boxes']}  <b>bbox_2d:</b> {esc(row['bbox_2d'])}</div>
      <div><b>scene:</b> {esc(row['scene'])}</div>
      <div><b>raw regions:</b><ul>{regions_html}</ul></div>
      <div><b>final relations:</b> {final_html} {"(fallback used)" if row['used_fallback'] else ""}</div>
      <div><b>edge-touch leak:</b> {edge_leak_html}</div>
    </div>
  </div>
  <div class="meta">id: {esc(row['id'])}</div>
</section>'''

    cards = "".join(build_card(r) for r in rows)
    doc = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>direction_method — zone_gen verification</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root {{ --bg:#f7f7f8; --card-bg:#fff; --text:#1a1a1a; --muted:#6b6b6b; --border:#e2e2e5; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#15161a; --card-bg:#1f2024; --text:#e8e8ea; --muted:#9a9aa2; --border:#33343a; }}
}}
* {{ box-sizing: border-box; }}
body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; background:var(--bg); color:var(--text); }}
main {{ max-width: 1000px; margin: 0 auto; padding: 20px 16px 60px; }}
h1 {{ font-size: 16px; }}
.card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 14px; margin-bottom: 16px; }}
.card-body {{ display: flex; gap: 16px; flex-wrap: wrap; }}
img {{ width: 260px; height: 260px; object-fit: cover; border-radius: 8px; flex-shrink: 0; border: 1px solid var(--border); }}
.info {{ font-size: 13px; flex: 1; min-width: 300px; line-height: 1.6; }}
ul {{ margin: 2px 0; padding-left: 18px; }}
.meta {{ font-size: 11px; color: var(--muted); margin-top: 8px; }}
</style>
</head>
<body>
<main>
<h1>{len(rows)} episodes — zone_gen locate + zones (seed={SAMPLE_SEED})</h1>
{cards}
</main>
</body>
</html>
'''
    GALLERY_PATH.parent.mkdir(parents=True, exist_ok=True)
    GALLERY_PATH.write_text(doc)


if __name__ == "__main__":
    main()
