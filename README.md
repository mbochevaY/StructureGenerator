# Structure generator

An open-source Streamlit tool for generating structure-varied connect-the-dots stimuli.

**S** is a formal structure score meant to track *perceived* structure (structure in vision as assessed by human subjects). Neither this generator nor the algorithm used to score S is explicitly sensitive to grouping principles such as closure, good continuation, or symmetry. Any Gestalt-flavored principles are a consequence (an emergent property) of the structure generation process.

Comparisons are controlled for grid size, number of line segments, and, as far as possible, perceptual similarity. Turtle-program length is reported alongside S. Any two stimuli can be compared on structure if they share the same grid size and number of line segments.

© 2026 Michaela Bocheva · Yale University

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run generate_examples_streamlit.py
```

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub (public).
2. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. **Create app** → this repo, branch `main`, main file `generate_examples_streamlit.py`.
4. Deploy. The live URL will look like `https://<app-name>.streamlit.app`.

## Files

| File | Role |
| --- | --- |
| `generate_examples_streamlit.py` | Interactive interface |
| `generate_examples.py` | Stimulus generation and S / turtle scoring |
| `.streamlit/config.toml` | Theme and static font serving |
| `static/latinmodern-math.otf` | Latin Modern Math (GUST license) |
