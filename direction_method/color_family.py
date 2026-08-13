"""Color-family reconciliation for self_check verdicts.

Why this exists, and why it is code rather than prompt text: `self_check` reads colors accurately
but will not apply the "these are the same family" rule no matter how it is written. Three separate
prompt attempts on 2026-08-11 -- counterexamples in the NOT-a-contradiction list, the same rule
moved into the contradiction rule where the decision is made, and the failing pairs quoted verbatim
-- each produced *identical* verdicts and identical evidence strings ("The shelves are clearly
white, not light grey"), 3/8 both before and after.

What the model does do reliably is *name what it sees*: every failing evidence string carries an
accurate color word. So the perception is taken from the model and the family judgment is made here,
which is the same move this project already made twice -- `checklist_update` stopped asking the model
which key an answer belonged under, and `context_parser` stopped trusting the model's own checklist.

Direction of effect: this only ever turns a `"no"` into a `"yes"`. It cannot invent a new
false-`"no"`; the worst case is a distractor that should have been rejected on color is not.

The two axes exist because a single flat list cannot work on this corpus. Mining all 1002 task
descriptions plus every oracle answer from the 2026-08-11 runs, the modifier is what decides the
family: `light brown` (186 uses) belongs with beige, `dark brown` (112) does not. Lightness is also
the *only* signal separating neutrals -- white and black are both "neutral" by hue -- while for
chromatic hues it is noise, since `navy`, `deep blue` and `light blue` are all the same blue.
"""

import re

# (hue, lightness). lightness: 0 light / 1 mid / 2 dark. Hue "neutral" covers the achromatic and
# warm-neutral run (white..black, plus wood and metals, which behave like neutrals in these scenes).
#
# Coverage is driven by the corpus, not by taste: every color word that occurs in
# episodes_train.jsonl or in the runs' oracle answers is here, and tests/test_color_family.py fails
# if a corpus word is missing. Entries beyond the corpus are common synonyms kept so an unseen
# eval set does not fall straight through.
COLOR_TERMS = {
    # --- neutral, light -------------------------------------------------------------------
    "white": ("neutral", 0), "off-white": ("neutral", 0), "offwhite": ("neutral", 0),
    "cream": ("neutral", 0), "creamy": ("neutral", 0), "ivory": ("neutral", 0),
    "eggshell": ("neutral", 0), "bone": ("neutral", 0), "chalk": ("neutral", 0),
    "beige": ("neutral", 0), "greige": ("neutral", 0), "sand": ("neutral", 0),
    "sandy": ("neutral", 0), "tan": ("neutral", 0), "taupe": ("neutral", 0),
    "khaki": ("neutral", 0), "oatmeal": ("neutral", 0), "linen": ("neutral", 0),
    "light grey": ("neutral", 0), "light gray": ("neutral", 0),
    "pale grey": ("neutral", 0), "pale gray": ("neutral", 0),
    "silver": ("neutral", 0), "silvery": ("neutral", 0), "chrome": ("neutral", 0),
    "stainless": ("neutral", 0), "stainless steel": ("neutral", 0), "steel": ("neutral", 0),
    "pewter": ("neutral", 0), "platinum": ("neutral", 0),
    "light wood": ("neutral", 0), "light wooden": ("neutral", 0), "blonde": ("neutral", 0),
    "blond": ("neutral", 0), "birch": ("neutral", 0), "ash": ("neutral", 0),
    "maple": ("neutral", 0), "pine": ("neutral", 0), "oak": ("neutral", 0),
    "light brown": ("neutral", 0), "light beige": ("neutral", 0), "light tan": ("neutral", 0),
    "marble": ("neutral", 0), "porcelain": ("neutral", 0),

    # --- neutral, mid ---------------------------------------------------------------------
    "grey": ("neutral", 1), "gray": ("neutral", 1), "slate": ("neutral", 1),
    "stone": ("neutral", 1), "concrete": ("neutral", 1), "granite": ("neutral", 1),
    "brown": ("neutral", 1), "wood": ("neutral", 1), "wooden": ("neutral", 1),
    "natural wood": ("neutral", 1), "timber": ("neutral", 1), "teak": ("neutral", 1),
    "honey": ("neutral", 1), "caramel": ("neutral", 1), "camel": ("neutral", 1),
    "bronze": ("neutral", 1), "brass": ("neutral", 1), "copper": ("neutral", 1),
    "metallic": ("neutral", 1), "gunmetal": ("neutral", 1),

    # --- neutral, dark --------------------------------------------------------------------
    "black": ("neutral", 2), "charcoal": ("neutral", 2), "ebony": ("neutral", 2),
    "dark grey": ("neutral", 2), "dark gray": ("neutral", 2),
    "dark brown": ("neutral", 2), "dark wood": ("neutral", 2), "dark wooden": ("neutral", 2),
    "dark bronze": ("neutral", 2), "walnut": ("neutral", 2), "mahogany": ("neutral", 2),
    "espresso": ("neutral", 2), "chocolate": ("neutral", 2), "coffee": ("neutral", 2),

    # --- chromatic: lightness recorded but not used to separate within a hue ---------------
    "blue": ("blue", 1), "navy": ("blue", 2), "dark blue": ("blue", 2),
    "deep blue": ("blue", 2), "light blue": ("blue", 0), "pale blue": ("blue", 0),
    "sky blue": ("blue", 0), "cobalt": ("blue", 1), "denim": ("blue", 1),
    "teal": ("blue", 1), "turquoise": ("blue", 1), "aqua": ("blue", 0), "cyan": ("blue", 0),
    "indigo": ("blue", 2), "powder blue": ("blue", 0),

    "green": ("green", 1), "dark green": ("green", 2), "light green": ("green", 0),
    "olive": ("green", 1), "sage": ("green", 0), "mint": ("green", 0),
    "emerald": ("green", 1), "forest": ("green", 2), "forest green": ("green", 2),
    "lime": ("green", 0), "jade": ("green", 1), "moss": ("green", 1),

    "red": ("red", 1), "dark red": ("red", 2), "bright red": ("red", 1),
    "crimson": ("red", 1), "scarlet": ("red", 1), "burgundy": ("red", 2),
    "maroon": ("red", 2), "wine": ("red", 2), "brick": ("red", 1),
    "terracotta": ("red", 1), "rust": ("red", 1), "salmon": ("red", 0),

    "orange": ("orange", 1), "peach": ("orange", 0), "apricot": ("orange", 0),
    "coral": ("orange", 0), "amber": ("orange", 1),

    "yellow": ("yellow", 1), "gold": ("yellow", 1), "golden": ("yellow", 1),
    "mustard": ("yellow", 1), "lemon": ("yellow", 0), "ochre": ("yellow", 1),

    "pink": ("pink", 0), "rose": ("pink", 0), "blush": ("pink", 0),
    "magenta": ("pink", 1), "fuchsia": ("pink", 1),

    "purple": ("purple", 1), "violet": ("purple", 1), "lilac": ("purple", 0),
    "lavender": ("purple", 0), "plum": ("purple", 2), "mauve": ("purple", 0),
}

# Not colors. "glass"/"clear" describe a material with no hue, and matching on them would let any
# two transparent things reconcile; they are excluded rather than mapped.
NON_COLOR_MATERIALS = frozenset({"glass", "clear", "transparent", "mirrored", "plastic", "fabric"})

# Longest first so "light grey" wins over "grey" and "dark blue" over "blue".
_TERMS_BY_LENGTH = sorted(COLOR_TERMS, key=len, reverse=True)
_TERM_RE = re.compile(
    r"(?<![a-z\-])(" + "|".join(re.escape(t) for t in _TERMS_BY_LENGTH) + r")(?![a-z])",
    re.IGNORECASE,
)


def colors_in(text: str) -> list:
    """Color terms present, longest-match-wins and de-overlapped, in order of appearance."""
    if not text:
        return []
    lowered = re.sub(r"[^a-z\- ]", " ", text.lower())
    found, spans = [], []
    for m in _TERM_RE.finditer(lowered):
        if any(s <= m.start() < e or s < m.end() <= e for s, e in spans):
            continue
        spans.append((m.start(), m.end()))
        if m.group(1) not in found:
            found.append(m.group(1))
    return found


def same_family(a: str, b: str) -> bool:
    """Whether two color terms are close enough that a difference between them is naming variance.

    Neutrals must agree on lightness exactly, since lightness is the only thing telling white from
    black. Allowing one step of slack was tried first and was too loose in exactly the place it
    matters: it reconciled "beige" with the bare "brown" in *"a dark, metallic brown, not beige"*
    (the comma keeps "dark brown" from matching as one term) and "black" with "brass", both of which
    are correct distractor rejections. Being strict here costs a few rescues and protects every
    rejection, which is the right direction for a mechanism that can only turn "no" into "yes".

    Chromatic hues match across their whole lightness range -- navy, deep blue and light blue are
    one blue -- and never match across hues.
    """
    if a not in COLOR_TERMS or b not in COLOR_TERMS:
        return False
    (hue_a, light_a), (hue_b, light_b) = COLOR_TERMS[a], COLOR_TERMS[b]
    if hue_a != hue_b:
        return False
    if hue_a == "neutral":
        return light_a == light_b
    return True


def reconcile(assertion: str, evidence: str) -> bool:
    """True when a `"no"` verdict should be read as a within-family color difference.

    Only called on a `"no"`. The evidence is typically of the form "X, not Y", restating the claimed
    color Y, so the claim's own terms are removed first -- what remains is what the model actually
    saw. If any perceived term is in the same family as any claimed term, the disagreement is naming
    variance, not a contradiction.
    """
    claimed = colors_in(assertion)
    if not claimed:
        return False
    perceived = [c for c in colors_in(evidence) if c not in claimed]
    if not perceived:
        return False
    return any(same_family(c, p) for c in claimed for p in perceived)
