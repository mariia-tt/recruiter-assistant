"""Draw the system data-flow diagram as a standalone SVG (docs/figures/architecture.svg)."""

from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "figures" / "architecture.svg"

STYLE = {  # fill, stroke
    "llm": ("#e8f1fc", "#2a78d6"),
    "rule": ("#f3f3f1", "#8a8984"),
    "data": ("#fff4dc", "#c98500"),
    "user": ("#ffffff", "#0b0b0b"),
}
INK, INK2 = "#0b0b0b", "#52514e"
W, H = 1180, 800
parts: list[str] = []


def box(x, y, w, h, title, sub="", kind="rule"):
    fill, stroke = STYLE[kind]
    rx = 18 if kind == "data" else 8
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>')
    ty = y + (h / 2 - 6 if sub else h / 2 + 5)
    parts.append(f'<text x="{x + w / 2}" y="{ty}" text-anchor="middle" font-size="14" '
                 f'font-weight="600" fill="{INK}">{title}</text>')
    for i, line in enumerate(sub.split("|") if sub else []):
        parts.append(f'<text x="{x + w / 2}" y="{ty + 18 + i * 15}" text-anchor="middle" '
                     f'font-size="11.5" fill="{INK2}">{line}</text>')


def arrow(points, label="", lx=None, ly=None, dashed=False):
    d = " ".join(f"{'M' if i == 0 else 'L'}{x},{y}" for i, (x, y) in enumerate(points))
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    parts.append(f'<path d="{d}" fill="none" stroke="{INK2}" stroke-width="1.4"{dash} '
                 f'marker-end="url(#arrow)"/>')
    if label:
        x = lx if lx is not None else (points[0][0] + points[-1][0]) / 2
        y = ly if ly is not None else (points[0][1] + points[-1][1]) / 2 - 6
        parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" font-size="11" '
                     f'fill="{INK2}">{label}</text>')


def lane(y, h, title):
    parts.append(f'<rect x="12" y="{y}" width="{W - 24}" height="{h}" rx="10" fill="none" '
                 f'stroke="#d6d5d0" stroke-width="1"/>')
    parts.append(f'<text x="28" y="{y + 22}" font-size="13" font-weight="700" '
                 f'fill="{INK2}">{title}</text>')


# ---- offline lane -------------------------------------------------------------
lane(14, 190, "OFFLINE — built once, cached")
box(40, 60, 170, 56, "Open datasets", "2 481 CVs · 1 759 labelled pairs", "data")
box(270, 60, 170, 56, "C0 Data cleaning", "PII mask · glued words · dedupe")
box(500, 44, 190, 56, "C2a Rule profiler", "skills dict · dates → years · level")
box(500, 128, 190, 56, "Chunk + embed + BM25", "bge-small-en · 200-word chunks")
box(760, 44, 180, 56, "Rule profiles", "for every CV", "data")
box(760, 128, 180, 56, "Hybrid index", "BM25 + dense vectors", "data")
arrow([(210, 88), (268, 88)])
arrow([(440, 80), (470, 80), (470, 72), (498, 72)])
arrow([(440, 96), (470, 96), (470, 156), (498, 156)])
arrow([(690, 72), (758, 72)])
arrow([(690, 156), (758, 156)])

# ---- online lane: search ------------------------------------------------------
lane(220, 300, "ONLINE — search (per vacancy)")
box(40, 300, 120, 56, "Recruiter", "", "user")
box(210, 300, 150, 56, "Streamlit UI", "chat · cards")
box(410, 262, 200, 92, "C1 Requirements", "collector|LLM: chat → JobSpec|rules: which question next", "llm")
box(660, 300, 180, 56, "C3 Hybrid retriever", "RRF fusion → top-50")
box(890, 300, 250, 56, "C4 Scorecard ranker", "7 weighted criteria + evidence")
box(890, 420, 250, 64, "C4b LLM assessment", "top-10 in one call · ratings 1–5|verified quotes", "llm")
arrow([(160, 328), (208, 328)])
arrow([(360, 318), (408, 318)], "message", ly=310)
arrow([(510, 354), (510, 390), (300, 390), (300, 358)], "follow-up question (≤ 2)", lx=405, ly=404)
arrow([(610, 318), (658, 318)], "JobSpec", ly=310)
arrow([(840, 328), (888, 328)], "top-50", ly=320)
arrow([(850, 184), (850, 250), (750, 250), (750, 298)], "", dashed=True)
arrow([(940, 100), (1010, 100), (1010, 298)], "", dashed=True)
arrow([(1015, 356), (1015, 418)], "top-10", lx=1045, ly=392)
arrow([(890, 452), (260, 452), (260, 358)], "ranked cards: score breakdown, has / missing skills, quotes",
      lx=590, ly=444)

# ---- online lane: outreach ----------------------------------------------------
lane(536, 210, "ONLINE — outreach (per selected candidate)")
box(210, 610, 150, 56, "“Write message”", "button on a card")
box(410, 610, 180, 56, "C2b LLM profiler", "roles · achievements", "llm")
box(640, 610, 180, 56, "C5 Outreach writer", "facts only · ≤ 110 words", "llm")
box(870, 588, 270, 100, "C6 Grounding checker", "LLM: claims + evidence quote|code: quote must be verbatim in CV|rules: length, placeholders, CTA", "llm")
arrow([(360, 638), (408, 638)])
arrow([(590, 638), (638, 638)], "facts", ly=630)
arrow([(820, 638), (868, 638)], "draft", ly=630)
arrow([(1005, 688), (1005, 712), (730, 712), (730, 668)], "unsupported claim → rewrite once", lx=870, ly=727)
arrow([(1005, 588), (1005, 572), (225, 572), (225, 358)], "verified message + ✓/✗ per claim", lx=590, ly=566)

# ---- legend -------------------------------------------------------------------
lx = 40
for kind, label in (("llm", "LLM call (Gemini, via llm.py: rate limit · cache · trace)"),
                    ("rule", "deterministic code"), ("data", "data / index")):
    fill, stroke = STYLE[kind]
    parts.append(f'<rect x="{lx}" y="762" width="14" height="14" rx="3" fill="{fill}" '
                 f'stroke="{stroke}" stroke-width="1.5"/>')
    parts.append(f'<text x="{lx + 20}" y="774" font-size="11.5" fill="{INK2}">{label}</text>')
    lx += 20 + len(label) * 6.2 + 24
parts.append(f'<text x="{lx}" y="774" font-size="11.5" fill="{INK2}">- - - reads from the '
             f'offline stores</text>')

svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}"
 font-family="Helvetica, Arial, sans-serif">
<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7"
 orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{INK2}"/></marker></defs>
<rect width="{W}" height="{H}" fill="#fcfcfb"/>
{chr(10).join(parts)}
</svg>
"""
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(svg)
print(f"wrote {OUT}")
