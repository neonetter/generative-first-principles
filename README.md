# Generative Modeling from First Principles

Interactive course materials on VAEs, GANs, and diffusion — math derived, then verified in code.

## Quick start (students)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Install Quarto: https://quarto.org/docs/download/
quarto render
open _book/index.html
```

To run the notebook live in Jupyter:

```bash
jupyter lab generative_first_principles.ipynb
```

## Project layout

| File | Purpose |
|------|---------|
| `index.qmd` | Landing page (setup, structure, tips) |
| `generative_first_principles.ipynb` | Main notebook |
| `build_notebook.py` | Regenerates the notebook from source |
| `_quarto.yml` | Book configuration |

## Rebuilding

```bash
python build_notebook.py   # optional: edit notebook source
quarto render              # build HTML book to _book/
quarto preview             # live preview with hot reload
```

Set `QUICK = False` in `build_notebook.py` (or in the notebook) for higher-fidelity training runs.

## GitHub Pages

The site publishes automatically on every push to `main` via [`.github/workflows/publish.yml`](.github/workflows/publish.yml).

**One-time setup:**

1. Create a GitHub repo named `generative-first-principles` (or update `site-path` in `_quarto.yml` if you use a different name).
2. Push this project to `main`:
   ```bash
   git remote add origin git@github.com:YOUR_USER/generative-first-principles.git
   git push -u origin main
   ```
3. In the repo **Settings → Actions → General**, set **Workflow permissions** to **Read and write**.
4. After the first workflow run, open **Settings → Pages** and confirm the source is the `gh-pages` branch (Quarto creates this automatically).

**Live site:** `https://YOUR_USER.github.io/generative-first-principles/`

**Manual publish** (optional, from your machine):

```bash
quarto publish gh-pages
```

The workflow uses `freeze: auto` and the notebook's cached outputs, so CI does not need PyTorch — renders stay fast.
