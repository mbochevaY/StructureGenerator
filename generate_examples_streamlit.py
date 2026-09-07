"""Interactive stimulus generator for generate_examples.py.

Run from this folder:

    streamlit run generate_examples_streamlit.py
"""

from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_examples as ge


st.set_page_config(
    page_title="Structure generator",
    layout="wide",
    initial_sidebar_state="expanded",
)
plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})

st.markdown(
    """
    <style>
      @font-face {
        font-family: "Latin Modern Math";
        src: url("app/static/latinmodern-math.otf") format("opentype");
        font-weight: 400;
        font-style: normal;
      }
      html, body, [class*="css"], .stApp, .stMarkdown, .stCaption, button, input, textarea, label {
        font-family: "Latin Modern Math", "Latin Modern Roman", Georgia, serif !important;
      }
      .block-container { padding-top: 1.1rem; padding-bottom: 3rem; }
      h1, h2, h3 {
        font-family: "Latin Modern Math", Georgia, serif !important;
        letter-spacing: -0.02em;
      }
      .brand-bar, .brand-foot {
        font-family: "Latin Modern Math", Georgia, serif !important;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #0B1220;
      }
      .brand-bar {
        font-size: 0.78rem;
        border-bottom: 1px solid #0B1220;
        padding: 0 0 0.55rem;
        margin: 0 0 1.1rem;
        display: flex;
        justify-content: space-between;
        gap: 1rem;
      }
      .brand-foot {
        font-size: 0.72rem;
        border-top: 1px solid #0B1220;
        padding: 0.85rem 0 0.2rem;
        margin-top: 2.2rem;
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        color: #111827;
      }
      .lede {
        max-width: 92%;
        width: 92%;
        font-size: 1.08rem;
        line-height: 1.55;
        color: #111827;
        margin: 0.15rem 0 1.1rem;
        font-family: "Latin Modern Math", Georgia, serif !important;
      }
      .lede p { margin: 0 0 0.55rem; }
      .metric-box {
        background: #ffffff;
        border: 1px solid #0B1220;
        border-radius: 0;
        padding: 0.75rem 0.9rem 0.8rem;
        margin-top: 0.45rem;
        font-family: "Latin Modern Math", Georgia, serif !important;
      }
      .metric-box .k {
        color: #000 !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        font-family: "Latin Modern Math", Georgia, serif !important;
      }
      .metric-box .v {
        color: #000 !important;
        font-size: 2.15rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.01em;
        line-height: 1.15;
        font-family: "Latin Modern Math", Georgia, serif !important;
      }
      div[data-testid="stSidebar"] {
        font-size: 1.22rem !important;
        min-width: 24rem;
      }
      div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 {
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-size: 1.12rem !important;
        font-family: "Latin Modern Math", Georgia, serif !important;
      }
      div[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
      div[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label,
      div[data-testid="stSidebar"] label p,
      div[data-testid="stSidebar"] label,
      div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
      div[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
      div[data-testid="stSidebar"] [data-baseweb="radio"] label,
      div[data-testid="stSidebar"] [data-baseweb="radio"] p,
      div[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
      div[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
      div[data-testid="stSidebar"] input,
      div[data-testid="stSidebar"] [data-baseweb="select"] *,
      div[data-testid="stSidebar"] [data-testid="stSliderTickBarMin"],
      div[data-testid="stSidebar"] [data-testid="stSliderTickBarMax"],
      div[data-testid="stSidebar"] [data-testid="stSliderThumbValue"] {
        font-size: 1.18rem !important;
        font-family: "Latin Modern Math", Georgia, serif !important;
      }
      div[data-testid="stSidebar"] [data-testid="stCaptionContainer"],
      div[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        font-size: 1.08rem !important;
      }
      div[data-testid="stButton"] button,
      div[data-testid="stButton"] button p,
      div[data-testid="stButton"] button span {
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        font-family: "Latin Modern Math", Georgia, serif !important;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def pad_schedule(values, n):
    if n <= 0:
        return []
    if len(values) >= n:
        return list(values[:n])
    return list(values) + [values[-1]] * (n - len(values))


def apply_knobs(knobs):
    ge.lineColor = knobs["line_color"]
    ge.lineWidth = knobs["line_width"]
    ge.gridColor = knobs["grid_color"]
    ge.levelFracs = knobs["level_fracs"]
    ge.acrossFracs = knobs["across_fracs"]
    ge.levelBias = knobs["level_bias"]
    ge.cellsPerLine = knobs["cells_per_line"]
    ge.gridCoarse = knobs["grid_coarse"]
    ge.gridN = knobs["grid_coarse"] * knobs["cells_per_line"]
    ge.minDiff = knobs["min_diff"]
    ge.connectBias = knobs["connect_bias"]
    ge.maxSpan = knobs["max_span"]


def shapes_by_line_count():
    grouped = defaultdict(list)
    seen = set()
    for name, _ in ge.latticeShapes(1):
        for u in (1, 2, 3):
            base = ge.baselineScene(name, u)
            if not ge.validScene(base) or not ge.fitsGrid(base):
                continue
            key = (name, u, len(base))
            if key in seen:
                continue
            seen.add(key)
            grouped[len(base)].append((name, u))
    return grouped


def format_program(cmds):
    return ge.formatTurtle(cmds)


def generate_examples(knobs):
    apply_knobs(knobs)
    rng = random.Random(knobs["seed"])
    levels = knobs["levels"]
    n_lines = knobs["n_lines"]
    n_examples = knobs["n_examples"]
    shape_name = knobs["shape"]
    mode = knobs["mode"]
    grouped = shapes_by_line_count()
    matches = list(grouped.get(n_lines, []))
    if shape_name != "Any":
        matches = [item for item in matches if item[0] == shape_name]
    if not matches:
        return [], "No baseline shape has that many line segments on the current grid."

    rng.shuffle(matches)
    examples = []
    error = None

    if mode == "Across categories":
        pool = ge.buildPool(levels, rng)
        filtered = defaultdict(list)
        for key, rows in pool.items():
            if key[1] != n_lines:
                continue
            if shape_name != "Any" and key[0] == 0:
                rows = [row for row in rows if row[2] == shape_name]
            filtered[key] = rows
        guard = 0
        while len(examples) < n_examples and guard < max(8, n_examples * 4):
            guard += 1
            anchor = None if shape_name == "Any" else shape_name
            result = ge.buildAcross(filtered, levels, rng, anchor=anchor)
            if result:
                examples.append(result)
        if not examples:
            error = "Could not assemble an across-category set with ascending S and turtle length. Try another seed or a milder pivot schedule."
        return examples, error

    if mode == "Objects":
        si = 0
        guard = 0
        while len(examples) < n_examples and guard < n_examples * len(matches) * 3:
            guard += 1
            name, u = matches[si % len(matches)]
            si += 1
            built = ge.buildObjects(name, u, rng)
            if built:
                examples.append((["chain", name, name], built[0], built[1], built[2]))
        if not examples:
            error = "Could not build an objects example. Try another seed or a different line count."
        return examples, error

    if mode == "Opened / misplaced":
        miss_pool = ge.buildMissPool(rng)
        anchors = [name for name, _ in matches]
        if shape_name != "Any":
            anchors = [shape_name]
        feasible = [anchor for anchor in anchors if ge.buildMissAcross(miss_pool, rng, 4000, anchor)]
        if not feasible:
            return [], "No opened/misplaced partners at this line count. Try a different line count or seed."
        rng.shuffle(feasible)
        for i in range(n_examples):
            result = ge.buildMissAcross(miss_pool, rng, 20000, feasible[i % len(feasible)])
            if result:
                examples.append(result)
        if not examples:
            error = "Could not assemble an opened/misplaced example."
        return examples, error

    si = 0
    guard = 0
    while len(examples) < n_examples and guard < n_examples * max(6, len(matches) * 3):
        guard += 1
        name, u = matches[si % len(matches)]
        si += 1
        built = ge.buildExample(name, u, levels, rng)
        if built:
            examples.append(([name] * levels, built[0], built[1], built[2]))
    if not examples:
        error = "Could not assemble a within-category example with strictly increasing S and turtle length. Try another seed, fewer levels, or a gentler pivot schedule."
    return examples, error


st.markdown(
    """
    <div class="brand-bar">
      <span>Michaela Bocheva</span>
      <span>Yale University</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.title("Structure generator")
st.markdown(
    """
    <div class="lede">
      <p>This interface generates structure-varied connect-the-dots stimuli. It varies a formal structure score, <b>S</b>, that is meant to track <i>perceived</i> structure: structure in vision as assessed by human subjects. Neither this generator nor the algorithm used to score S is explicitly sensitive to grouping principles such as closure, good continuation, or symmetry. Rather, any Gestalt-flavored principles are a consequence (i.e., an emergent property) of the structure generation process.</p>
      <p>Comparisons are controlled for things like number of line segments, grid size, and, as far as possible, perceptual similarity. Turtle-program length is reported alongside S as a second complexity measure. Any two stimuli can be compared in terms of their structure as long as they share the same grid size and number of line segments.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

sidebar = st.sidebar
sidebar.header("Generation")

mode = sidebar.radio(
    "Assembly mode",
    ("Within category", "Across categories", "Objects", "Opened / misplaced"),
    help="Within category warps one closed family across levels. Across categories mixes different families at the same line count. Objects uses a chain, a lightly pivoted shape, then a top-level warp. Opened / misplaced unfolds or displaces a closed outline.",
)
if "seed" not in st.session_state:
    st.session_state.seed = 0
seed = sidebar.number_input(
    "Random seed",
    min_value=0,
    max_value=10_000,
    step=1,
    key="seed",
    help="Fixes the random choices used to pivot lines and assemble a set. Change this to get a new example with the same knobs.",
)
default_levels = 2 if mode == "Opened / misplaced" else 3
levels = sidebar.slider(
    "Levels",
    min_value=2,
    max_value=5,
    value=default_levels,
    help="How many figures to generate in one example. Consecutive levels must increase both S and turtle length.",
    disabled=mode in {"Objects", "Opened / misplaced"},
)
if mode == "Opened / misplaced":
    levels = 2
elif mode == "Objects":
    levels = 3

n_examples = sidebar.slider(
    "Examples",
    min_value=1,
    max_value=6,
    value=1,
    help="How many independent example sets to generate with the current knobs.",
)

sidebar.header("Grid and lines")
grid_coarse = sidebar.slider(
    "Grid size (coarse bands)",
    min_value=3,
    max_value=8,
    value=5,
    help="Number of coarse bands along each side of the figure. The drawn dot grid is this value times the cells-per-line setting. Larger grids allow bigger shapes.",
)
cells_per_line = sidebar.slider(
    "Cells per line",
    min_value=2,
    max_value=10,
    value=5,
    help="The universal line unit: every line spans this many fine-grid cells and therefore connects this many + 1 dots. This is the length of one turtle 'Draw line' step on the fine grid.",
)
max_span = sidebar.slider(
    "Maximum shape span",
    min_value=2,
    max_value=grid_coarse,
    value=max(2, grid_coarse - 1),
    help="Hard cap on a figure's width or height, measured in coarse bands. By default this is one less than the coarse grid so figures stay inside the dotted field.",
)

ge.gridCoarse = grid_coarse
ge.cellsPerLine = cells_per_line
ge.gridN = grid_coarse * cells_per_line
ge.maxSpan = max_span

available = shapes_by_line_count()
line_counts = sorted(available)
if not line_counts:
    sidebar.error("No shapes fit this grid. Increase grid size or maximum span.")
    st.stop()

n_lines = sidebar.select_slider(
    "Number of line segments",
    options=line_counts,
    value=4 if 4 in line_counts else line_counts[0],
    help="How many unit line segments each figure contains. Every level in a set keeps this count. Available values come from closed library shapes that fit the current grid.",
)
shape_options = ["Any"] + sorted({name for name, _ in available[n_lines]})
shape = sidebar.selectbox(
    "Baseline shape",
    shape_options,
    help="Closed family used as the starting figure. Any picks among families that have the chosen line count. Across-category mode uses this as the Level 1 anchor when it is not Any.",
)

sidebar.caption(
    f"Fine grid: **{grid_coarse * cells_per_line} × {grid_coarse * cells_per_line}** dots. "
    f"Each line covers **{cells_per_line}** cells."
)

sidebar.header("Pivot schedule")
level_fracs = []
default_fracs = pad_schedule([0.0, 0.3, 0.6], levels)
for i in range(levels):
    level_fracs.append(
        sidebar.slider(
            f"Level {i + 1} pivot fraction",
            min_value=0.0,
            max_value=1.0,
            value=float(default_fracs[i]),
            step=0.05,
            help="Fraction of lines that swing about a shared vertex at this level. Level 1 is usually 0 so the closed baseline stays intact.",
        )
    )

across_fracs = list(ge.acrossFracs)
if mode == "Across categories":
    sidebar.subheader("Across-category schedule")
    across_fracs = []
    default_across = pad_schedule([0.0, 0.2, 0.4], levels)
    for i in range(levels):
        across_fracs.append(
            sidebar.slider(
                f"Across level {i + 1} pivot fraction",
                min_value=0.0,
                max_value=1.0,
                value=float(default_across[i]),
                step=0.05,
                help="Gentler pivot schedule used when each level comes from a different shape family.",
            )
        )

connect_bias = sidebar.slider(
    "Connect bias",
    min_value=0.0,
    max_value=1.0,
    value=0.9,
    step=0.05,
    help="Chance that a pivoted line tries to land on an existing dot. Higher values keep the drawing in one piece instead of scattering isolated sticks.",
)
min_diff = sidebar.slider(
    "Minimum orientation change",
    min_value=0.0,
    max_value=1.0,
    value=0.1,
    step=0.05,
    help="How different consecutive levels must look. 0 allows the same mix of line orientations; 1 requires no shared orientations.",
)

with sidebar.expander("Per-level attach bias", expanded=False):
    st.caption("After pivoting, this is the chance the swung line stays attached to the rest of the figure. Lower values make that level more irregular.")
    level_bias = []
    default_bias = pad_schedule([0.9, 0.9, 0.85], levels)
    for i in range(levels):
        level_bias.append(
            st.slider(
                f"Level {i + 1} attach bias",
                min_value=0.0,
                max_value=1.0,
                value=float(default_bias[i]),
                step=0.05,
                help="Stay-attached bias used when mutating this level.",
            )
        )

sidebar.header("Appearance")
line_color = sidebar.color_picker(
    "Line color",
    value="#000000",
    help="Stroke and endpoint color for the figure.",
)
line_width = sidebar.slider(
    "Line width",
    min_value=1.0,
    max_value=12.0,
    value=5.5,
    step=0.5,
    help="Thickness of each line segment in the preview.",
)
grid_color = sidebar.color_picker(
    "Grid-dot color",
    value="#D1D1D1",
    help="Color of the background lattice dots.",
)

knobs = dict(
    mode=mode,
    seed=int(seed),
    levels=int(levels),
    n_examples=int(n_examples),
    grid_coarse=int(grid_coarse),
    cells_per_line=int(cells_per_line),
    max_span=int(max_span),
    n_lines=int(n_lines),
    shape=shape,
    level_fracs=level_fracs,
    across_fracs=across_fracs,
    level_bias=level_bias,
    connect_bias=float(connect_bias),
    min_diff=float(min_diff),
    line_color=line_color,
    line_width=float(line_width),
    grid_color=grid_color,
)
apply_knobs(knobs)

def bump_seed():
    st.session_state.seed = int(st.session_state.get("seed", 0)) + 1
    st.session_state.force_generate = True


col_gen, col_again, _ = st.columns([1, 1, 3])
with col_gen:
    generate_clicked = st.button("Generate", type="primary", width="stretch")
with col_again:
    st.button("New seed", on_click=bump_seed, width="stretch")

if generate_clicked or st.session_state.pop("force_generate", False) or "examples" not in st.session_state:
    knobs["seed"] = int(st.session_state.seed)
    with st.spinner("Generating stimuli…"):
        examples, error = generate_examples(knobs)
    st.session_state.examples = examples
    st.session_state.error = error
    st.session_state.generated_knobs = knobs

st.caption("Appearance knobs update the preview immediately. Grid, line count, and pivot knobs apply when you generate.")

examples = st.session_state.get("examples") or []
error = st.session_state.get("error")
apply_knobs(knobs)

if error and not examples:
    st.warning(error)
elif not examples:
    st.info("Set the knobs, then click Generate.")
else:
    if error:
        st.info(error)
    for example_i, (names, scenes, counts, turtles) in enumerate(examples, start=1):
        if len(examples) > 1:
            st.subheader(f"Example {example_i}")
        columns = st.columns(len(scenes))
        for level_i, (column, name, scene, n_rand, turtle) in enumerate(
            zip(columns, names, scenes, counts, turtles), start=1
        ):
            s_value = ge.structureValue(scene)
            with column:
                st.markdown(f"**Level {level_i}** · {name}")
                fig = ge.stimulusFigure(scene, size=3.0, full_grid=True)
                st.pyplot(fig, width="content")
                plt.close(fig)
                st.markdown(
                    f'<div class="metric-box">'
                    f'<div class="k" style="color:#000;font-size:1.35rem;font-weight:700">S</div>'
                    f'<div class="v" style="color:#000;font-size:2.15rem;font-weight:800">{s_value}</div>'
                    f'<div class="k" style="color:#000;font-size:1.35rem;font-weight:700;margin-top:0.45rem">Turtle length</div>'
                    f'<div class="v" style="color:#000;font-size:2.15rem;font-weight:800">{turtle}</div>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"{len(scene)} line segments · {ge.dotCount(scene)} dots · "
                    f"{n_rand} pivoted"
                )
                cmds, length = ge.turtleProgram(scene)
                with st.expander("Minimal turtle program", expanded=True):
                    if cmds:
                        st.code("\n".join(format_program(cmds)), language=None)
                        st.caption(f"Length {length}: command count, with Repeat = 1 + body.")
                    else:
                        st.write("No program.")
        st.divider()

st.markdown(
    """
    <div class="brand-foot">
      <span>© 2026 Michaela Bocheva</span>
      <span>Yale University</span>
    </div>
    """,
    unsafe_allow_html=True,
)
