"""Build a single self-contained HTML failure report from a sweep's episode logs.

Why this exists: `serve_viewer.py` is a localhost server, so it shows the traces to whoever is
sitting on the GPU box and nobody else. This produces one HTML file with every image inlined as a
data URI and no external requests at all, so it can be mailed, committed, or published and opened
by anyone.

Only FAILED runs are included -- the successes are not what anyone reviews, and dropping them is
what keeps the file inside a publishable size budget.

Usage:
    python direction_method/scripts/build_failure_report.py \
        --log-root /path/to/sweep_logs --out failures.html --title "qwen3.6 v2"
"""

import argparse
import base64
import html
import io
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

# Roughly 12 KB of base64 per image at these settings; the budget check below reports the real
# total so the caller can lower them if a sweep has an unusually large number of failures.
THUMB_PX = 340
JPEG_QUALITY = 62


def encode_image(path: Path) -> str | None:
    try:
        im = Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None
    im.thumbnail((THUMB_PX, THUMB_PX))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=JPEG_QUALITY, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def classify(reasoning: str) -> str:
    """Group failures by which check actually ended the episode. Derived from the questioner's own
    reasoning string, so it stays truthful to the code rather than inventing a taxonomy."""
    r = (reasoning or "").lower()
    if "checklist self_check failed" in r:
        return "checklist rejected candidate"
    if "self_check failed on answer" in r:
        return "answer rejected candidate"
    if "all relations verified" in r:
        return "accepted (nothing left to ask)"
    if "budget" in r:
        return "budget exhausted"
    if "zone_gen" in r:
        return "zone_gen error"
    return "other"


def relation_of(reasoning: str) -> str:
    m = re.search(r"relation '([^']+)'", reasoning or "") or re.search(r"failed on '([^']+)'", reasoning or "")
    return m.group(1) if m else ""


def number_and_assign(records: list, names: list) -> None:
    """Number the cases 1..N in list order, then hand out contiguous blocks.

    Contiguous, not stratified: the reviewers asked for a straight cut so each person's range is a
    single span of case numbers ("I have 82 through 162"), which is easier to talk about and to
    split further than an interleaved deal. The trade-off is that each person's slice is not a
    proportional mix of description types -- fine here, since the numbers are for dividing reading
    work, not for per-reviewer statistics.

    Any remainder goes to the earliest reviewers, so block sizes differ by at most one.
    Deterministic for a given log directory and name order, so rebuilding does not reshuffle work
    someone has already started.
    """
    for i, r in enumerate(records, start=1):
        r["num"] = i
    if not names:
        return
    n, k = len(records), len(names)
    base, extra = divmod(n, k)
    start = 0
    for i, name in enumerate(names):
        size = base + (1 if i < extra else 0)
        for r in records[start:start + size]:
            r["assignee"] = name
        start += size


def collect(log_root: Path) -> tuple[list, dict]:
    episodes_dir = log_root / "episodes"
    if not episodes_dir.is_dir():
        raise SystemExit(f"no episodes/ under {log_root}")

    records, totals = [], Counter()
    for path in sorted(episodes_dir.glob("*.json")):
        d = json.loads(path.read_text())
        totals["runs"] += 1
        if d.get("full_success"):
            totals["passed"] += 1
            continue
        totals["failed"] += 1

        cands = d.get("candidates") or []
        last = cands[-1] if cands else {}
        kind = classify(last.get("reasoning", ""))
        totals[f"kind:{kind}"] += 1

        shown = []
        for c in cands:
            img = None
            p = c.get("boxed_image_path") or c.get("raw_image_path")
            if p:
                p = Path(p)
                if not p.is_absolute():
                    p = log_root / p
                if p.exists():
                    img = encode_image(p)
            shown.append({
                "index": c.get("index"),
                "img": img,
                "bbox": c.get("bbox_2d"),
                "zones": c.get("zone_list") or [],
                "scene": c.get("scene", ""),
                "conclusion": c.get("conclusion"),
                "reasoning": c.get("reasoning", ""),
                "checklist_before": c.get("checklist_before") or {},
                "interactions": c.get("interactions") or [],
            })

        cp = d.get("context_parser") or {}
        records.append({
            "id": d.get("episode_id"),
            "idx": d.get("episode_idx"),
            "type": d.get("task_type"),
            "category": d.get("category"),
            "desc": d.get("target_description"),
            "kind": kind,
            "relation": relation_of(last.get("reasoning", "")),
            "outcome": d.get("outcome"),
            "n_dist": d.get("n_distractors"),
            "n_succ": d.get("n_successes"),
            "n_q": d.get("n_questions"),
            "target_category": cp.get("target_category", ""),
            "target_phrase": cp.get("target_phrase", ""),
            "candidates": shown,
        })
    return records, totals


PAGE = """<title>{title_esc}</title>
<style>
:root {{
  color-scheme: light dark;
  --bg:#f6f7f9; --panel:#ffffff; --ink:#15181d; --muted:#5b6472; --line:#dfe3ea;
  --accent:#37618f; --accent-soft:#e8eef6;
  --pass:#2c7157; --fail:#a8402d; --warn:#8a6420;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
  --sans:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",sans-serif;
}}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#0f1216; --panel:#161b21; --ink:#e4e8ee; --muted:#98a2b1; --line:#262d36;
           --accent:#7aa7d9; --accent-soft:#1b2531;
           --pass:#5cbf95; --fail:#e0806c; --warn:#d2a75a; }}
}}
:root[data-theme="dark"] {{ --bg:#0f1216; --panel:#161b21; --ink:#e4e8ee; --muted:#98a2b1; --line:#262d36;
  --accent:#7aa7d9; --accent-soft:#1b2531; --pass:#5cbf95; --fail:#e0806c; --warn:#d2a75a; }}
:root[data-theme="light"] {{ --bg:#f6f7f9; --panel:#ffffff; --ink:#15181d; --muted:#5b6472; --line:#dfe3ea;
  --accent:#37618f; --accent-soft:#e8eef6; --pass:#2c7157; --fail:#a8402d; --warn:#8a6420; }}

* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--ink); font-family:var(--sans);
        font-size:14px; line-height:1.5; }}
h1 {{ font-size:17px; margin:0; font-weight:640; letter-spacing:-0.01em; }}
.eyebrow {{ font-size:11px; letter-spacing:0.09em; text-transform:uppercase; color:var(--muted); font-weight:600; }}

header {{ padding:16px 20px; border-bottom:1px solid var(--line); background:var(--panel);
          display:flex; flex-wrap:wrap; gap:16px 28px; align-items:baseline; }}
.stat {{ display:flex; flex-direction:column; gap:2px; }}
.stat b {{ font-size:19px; font-variant-numeric:tabular-nums; font-weight:660; }}

.filters {{ display:flex; flex-wrap:wrap; gap:8px; padding:12px 20px;
            border-bottom:1px solid var(--line); background:var(--panel); }}
select, input {{ font:inherit; color:inherit; background:var(--bg); border:1px solid var(--line);
                 border-radius:6px; padding:6px 9px; }}
input {{ min-width:200px; flex:1; }}
:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}

main {{ display:grid; grid-template-columns:minmax(280px,360px) 1fr; height:calc(100vh - 116px); }}
@media (max-width:840px) {{ main {{ grid-template-columns:1fr; height:auto; }} #detail {{ border-left:0; }} }}

#list {{ overflow-y:auto; border-right:1px solid var(--line); }}
.row {{ display:block; width:100%; text-align:left; background:none; border:0; border-bottom:1px solid var(--line);
        padding:10px 14px; cursor:pointer; color:inherit; font:inherit; }}
.row:hover {{ background:var(--accent-soft); }}
.row[aria-current="true"] {{ background:var(--accent-soft); box-shadow:inset 3px 0 0 var(--accent); }}
.row .d {{ font-weight:600; }}
.row .m {{ color:var(--muted); font-size:12px; display:flex; gap:8px; flex-wrap:wrap; margin-top:2px; }}

#detail {{ overflow-y:auto; padding:20px 24px; }}
.chip {{ display:inline-block; font-size:11px; padding:1px 7px; border-radius:999px;
         border:1px solid var(--line); color:var(--muted); white-space:nowrap; }}
.chip.fail {{ color:var(--fail); border-color:currentColor; }}
.chip.pass {{ color:var(--pass); border-color:currentColor; }}
.chip.warn {{ color:var(--warn); border-color:currentColor; }}

.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px;
         padding:16px; margin-bottom:16px; }}
.card h3 {{ margin:0 0 10px; font-size:13px; letter-spacing:0.04em; text-transform:uppercase; color:var(--muted); }}
.kv {{ display:grid; grid-template-columns:auto 1fr; gap:4px 14px; }}
.kv dt {{ color:var(--muted); font-size:12px; }}
.kv dd {{ margin:0; font-family:var(--mono); font-size:12.5px; word-break:break-word; }}
figure {{ margin:0 0 12px; }}
figure img {{ max-width:100%; border-radius:8px; border:1px solid var(--line); display:block; }}
figcaption {{ font-size:11.5px; color:var(--muted); margin-top:5px; }}

.step {{ border-left:2px solid var(--line); padding:8px 0 8px 12px; margin-bottom:2px; }}
.step.bad {{ border-left-color:var(--fail); }}
.step .top {{ display:flex; gap:8px; align-items:baseline; flex-wrap:wrap; }}
.step .q, .step .a, .step .e {{ font-family:var(--mono); font-size:12.5px; margin-top:3px; }}
.step .a {{ color:var(--ink); }}
.step .e {{ color:var(--muted); }}
.lbl {{ font-size:10.5px; letter-spacing:0.07em; text-transform:uppercase; color:var(--muted); }}
.empty {{ color:var(--muted); padding:40px 0; text-align:center; }}
code {{ font-family:var(--mono); }}

/* Reviewer badge. Three hues held apart in lightness as well as hue, so the owner of a row is
   readable at a glance and still distinguishable without relying on color alone (the name is
   written out, the tint only speeds up scanning). */
/* Case number. Tabular figures and a fixed width so the numbers form a clean column the eye can
   run down, and so "I'm on 137" is findable by scrolling rather than searching. */
.num {{ display:inline-block; min-width:3.2ch; margin-right:7px; text-align:right;
        font-variant-numeric:tabular-nums; font-size:12px; font-weight:600; color:var(--muted); }}
.who {{ display:inline-block; font-size:10.5px; font-weight:680; letter-spacing:0.04em;
        padding:1px 6px; border-radius:4px; margin-right:7px; vertical-align:1px;
        background:var(--who-bg,var(--accent-soft)); color:var(--who-fg,var(--accent)); }}
{who_css}
</style>

<header>
  <div class="stat"><span class="eyebrow">{title_esc}</span><b>{n_fail}</b><span class="eyebrow">failed runs</span></div>
  <div class="stat"><span class="eyebrow">of</span><b>{n_runs}</b><span class="eyebrow">total runs</span></div>
  <div class="stat"><span class="eyebrow">pass rate</span><b>{pass_rate}</b><span class="eyebrow">full success</span></div>
  {who_stats}
  <div style="flex:1"></div>
  <div style="max-width:46ch; color:var(--muted); font-size:12.5px;">
    Every failed run, with the oracle exchange that ended it. Monospace text is verbatim model
    output. The step outlined in red is the check that rejected the candidate.
  </div>
</header>

<div class="filters">
  {who_select}
  <select id="f-type"><option value="">description type: all</option>{type_opts}</select>
  <select id="f-kind"><option value="">failure mode: all</option>{kind_opts}</select>
  <input id="f-q" type="search" placeholder="search description, category, target phrase" />
  <span id="count" class="chip"></span>
</div>

<main>
  <div id="list"></div>
  <div id="detail"><p class="empty">Select a run on the left.</p></div>
</main>

<script>
const DATA = {data_json};
const list = document.getElementById('list');
const detail = document.getElementById('detail');
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}}[c]));
let shown = [], current = -1;

function matches(r) {{
  const whoEl = document.getElementById('f-who');
  const t = document.getElementById('f-type').value;
  const k = document.getElementById('f-kind').value;
  const q = document.getElementById('f-q').value.trim().toLowerCase();
  if (whoEl && whoEl.value && r.assignee !== whoEl.value) return false;
  if (t && r.type !== t) return false;
  if (k && r.kind !== k) return false;
  if (q && ![r.desc, r.category, r.target_phrase, r.target_category].join(' ').toLowerCase().includes(q)) return false;
  return true;
}}

function renderList() {{
  shown = DATA.filter(matches);
  document.getElementById('count').textContent = shown.length + ' shown';
  list.innerHTML = shown.map((r, i) => `
    <button class="row" data-i="${{i}}">
      <div class="d"><span class="num">${{r.num}}</span>${{r.assignee ? `<span class="who who-${{esc(r.assignee)}}">${{esc(r.assignee)}}</span>` : ''}}${{esc(r.desc)}}</div>
      <div class="m">
        <span class="chip">${{esc(r.type)}}</span>
        <span class="chip fail">${{esc(r.kind)}}${{r.relation ? ' · ' + esc(r.relation) : ''}}</span>
        <span>${{r.n_succ}}/${{r.n_dist}} correct · ${{r.n_q}} questions</span>
      </div>
    </button>`).join('') || '<p class="empty">No runs match these filters.</p>';
  current = -1;
  detail.innerHTML = '<p class="empty">Select a run on the left.</p>';
}}

function stepHTML(it, isLast) {{
  const bad = it.verdict === 'no';
  const region = it.region ? `<span class="chip">${{esc(it.region)}}</span>` : '';
  const key = esc(it.relation || it.parent_key || '');
  const claim = it.type === 'checklist_check'
    ? `<div class="a"><span class="lbl">claim</span> ${{esc(it.assertion)}}</div>`
    : `<div class="q"><span class="lbl">asked</span> ${{esc(it.question)}}</div>
       <div class="a"><span class="lbl">oracle</span> ${{esc(it.answer)}}</div>`;
  return `<div class="step ${{bad ? 'bad' : ''}}">
    <div class="top">
      <span class="chip ${{bad ? 'fail' : 'pass'}}">${{bad ? 'contradicted' : 'consistent'}}</span>
      <span class="chip">${{key}}</span>${{region}}
      <span class="lbl">${{it.type === 'checklist_check' ? 'checklist re-check' : 'new question'}}</span>
    </div>
    ${{claim}}
    <div class="e"><span class="lbl">evidence</span> ${{esc(it.evidence)}}</div>
  </div>`;
}}

function renderDetail(i) {{
  const r = shown[i];
  if (!r) return;
  current = i;
  [...list.querySelectorAll('.row')].forEach((b, j) => b.setAttribute('aria-current', j === i));

  const head = `<div class="card">
    <h3>case ${{r.num}}${{r.assignee ? ` — reviewer <span class="who who-${{esc(r.assignee)}}">${{esc(r.assignee)}}</span>` : ''}}</h3>
    <dl class="kv">
      <dt>description</dt><dd>${{esc(r.desc)}}</dd>
      <dt>true category</dt><dd>${{esc(r.category)}}</dd>
      <dt>parsed target</dt><dd>${{esc(r.target_phrase)}} <span class="chip">${{esc(r.target_category)}}</span></dd>
      <dt>type</dt><dd>${{esc(r.type)}}</dd>
      <dt>result</dt><dd>${{r.n_succ}} of ${{r.n_dist}} candidates correct, ${{r.n_q}} questions</dd>
      <dt>ended by</dt><dd>${{esc(r.kind)}}${{r.relation ? ' — ' + esc(r.relation) : ''}}</dd>
    </dl>
  </div>`;

  const cards = r.candidates.map((c, ci) => {{
    const img = c.img
      ? `<figure><img src="${{c.img}}" alt="candidate ${{ci}} with the target boxed" />
         <figcaption>candidate ${{ci}} · red box = grounded target · zones asked: ${{esc((c.zones || []).join(', ') || 'none')}}</figcaption></figure>`
      : '<p class="empty" style="padding:12px 0">image not available</p>';
    const before = Object.keys(c.checklist_before || {{}}).length
      ? `<dl class="kv">` + Object.entries(c.checklist_before).map(([k, v]) =>
          `<dt>${{esc(k)}}</dt><dd>${{v.map(esc).join('<br>')}}</dd>`).join('') + `</dl>`
      : '<p style="color:var(--muted);font-size:12.5px;margin:0">empty — nothing carried in yet</p>';
    return `<div class="card">
      <h3>candidate ${{ci}} — concluded ${{c.conclusion ? 'match' : 'mismatch'}}</h3>
      ${{img}}
      <h3 style="margin-top:14px">checklist on arrival</h3>
      ${{before}}
      <h3 style="margin-top:14px">checks, in order</h3>
      ${{(c.interactions || []).map((it, k) => stepHTML(it, k === c.interactions.length - 1)).join('') || '<p style="color:var(--muted)">none</p>'}}
      ${{c.scene ? `<h3 style="margin-top:14px">zone_gen scene</h3><p class="e" style="font-family:var(--mono);font-size:12.5px;margin:0">${{esc(c.scene)}}</p>` : ''}}
    </div>`;
  }}).join('');

  detail.innerHTML = head + cards;
  detail.scrollTop = 0;
}}

list.addEventListener('click', e => {{
  const b = e.target.closest('.row');
  if (b) renderDetail(Number(b.dataset.i));
}});
// f-who included here: it was omitted once, and the reviewer dropdown then rendered but did
// nothing when changed. Guarded because the element only exists when --assignees was passed.
['f-who', 'f-type', 'f-kind', 'f-q'].forEach(id => {{
  const el = document.getElementById(id);
  if (el) el.addEventListener('input', renderList);
}});
renderList();
</script>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="direction_method failures")
    ap.add_argument(
        "--assignees", default="",
        help="Comma-separated reviewer names. Cases are split evenly between them, stratified by "
             "description type and failure mode so nobody gets a skewed slice.",
    )
    args = ap.parse_args()

    root = Path(args.log_root)
    records, totals = collect(root)
    if not records:
        raise SystemExit("no failed runs found — nothing to report")

    names = [n.strip() for n in args.assignees.split(",") if n.strip()]
    number_and_assign(records, names)

    kinds = sorted({r["kind"] for r in records})
    types = sorted({r["type"] for r in records})
    runs, failed = totals["runs"], totals["failed"]

    # Tints are assigned by position, so the palette is stable for a given name order.
    TINTS = [("#e4edf7", "#2c5a8a"), ("#e6f1ea", "#256b4f"), ("#f4ebe2", "#8a5a2b")]
    TINTS_DARK = [("#1d2a38", "#8fb6e0"), ("#1b2b25", "#6cc39c"), ("#302620", "#d4a273")]
    who_css = who_select = who_stats = ""
    if names:
        per = Counter(r["assignee"] for r in records)
        rules = []
        for i, n in enumerate(names):
            bg, fg = TINTS[i % len(TINTS)]
            dbg, dfg = TINTS_DARK[i % len(TINTS_DARK)]
            sel = f'.who-{html.escape(n)}'
            rules.append(f"{sel} {{ --who-bg:{bg}; --who-fg:{fg}; }}")
            rules.append(f'@media (prefers-color-scheme: dark) {{ {sel} {{ --who-bg:{dbg}; --who-fg:{dfg}; }} }}')
            rules.append(f':root[data-theme="dark"] {sel} {{ --who-bg:{dbg}; --who-fg:{dfg}; }}')
            rules.append(f':root[data-theme="light"] {sel} {{ --who-bg:{bg}; --who-fg:{fg}; }}')
        who_css = "\n".join(rules)
        # Each reviewer owns one contiguous span, so show it -- "Jaemin 82-162" is the useful fact.
        spans = {}
        for n in names:
            nums = [r["num"] for r in records if r.get("assignee") == n]
            spans[n] = (min(nums), max(nums)) if nums else (0, 0)
        who_select = (
            '<select id="f-who"><option value="">reviewer: everyone</option>'
            + "".join(
                f'<option value="{html.escape(n)}">{html.escape(n)} — {spans[n][0]}–{spans[n][1]} ({per[n]})</option>'
                for n in names
            )
            + "</select>"
        )
        who_stats = "".join(
            f'<div class="stat"><span class="eyebrow"><span class="who who-{html.escape(n)}">{html.escape(n)}</span></span>'
            f'<b>{spans[n][0]}–{spans[n][1]}</b><span class="eyebrow">{per[n]} cases</span></div>'
            for n in names
        )

    page = PAGE.format(
        who_css=who_css,
        who_select=who_select,
        who_stats=who_stats,
        title_esc=html.escape(args.title),
        n_fail=failed,
        n_runs=runs,
        pass_rate=f"{100 * (runs - failed) / runs:.1f}%" if runs else "—",
        type_opts="".join(f'<option value="{html.escape(t)}">{html.escape(t)}</option>' for t in types),
        kind_opts="".join(f'<option value="{html.escape(k)}">{html.escape(k)}</option>' for k in kinds),
        data_json=json.dumps(records, ensure_ascii=False),
    )

    out = Path(args.out)
    out.write_text(page, encoding="utf-8")
    mb = out.stat().st_size / 1024 / 1024
    print(f"{out}  {mb:.1f} MB  ({failed} failed of {runs} runs)")
    for k, v in sorted(totals.items()):
        if k.startswith("kind:"):
            print(f"  {k[5:]}: {v}")
    if mb > 15:
        print("WARNING: over 15 MB -- lower THUMB_PX or JPEG_QUALITY before publishing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
