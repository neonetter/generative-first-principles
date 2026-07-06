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
quarto render              # HTML + PDF to _book/
quarto preview             # live preview with hot reload
```

### PDF

PDF needs a LaTeX engine. **TinyTeX is enough** — you do not need the full MacTeX install (~4 GB):

```bash
quarto install tinytex      # one-time, ~100 MB
quarto render --to pdf      # or: quarto render (builds HTML + PDF)
open _book/Generative-Modeling-from-First-Principles.pdf
```

The GitHub Actions workflow installs TinyTeX automatically; the PDF is published alongside the HTML book.

Set `QUICK = False` in `build_notebook.py` (or in the notebook) for higher-fidelity training runs.

## GitHub Pages

The site publishes automatically on every push to `main` via [`.github/workflows/publish.yml`](.github/workflows/publish.yml).

**Live site:** https://neonetter.github.io/generative-first-principles/

**PDF:** https://neonetter.github.io/generative-first-principles/Generative-Modeling-from-First-Principles.pdf

**One-time setup (already done for this repo):**

1. Push to `main` on `git@github.com:neonetter/generative-first-principles.git`
2. In the repo **Settings → Actions → General**, set **Workflow permissions** to **Read and write**
3. After the first workflow run, open **Settings → Pages** and confirm the source is the `gh-pages` branch

**Manual publish** (optional, from your machine):

```bash
quarto publish gh-pages
```

The workflow uses `freeze: auto` and the notebook's cached outputs, so CI does not need PyTorch — renders stay fast.

**Georgia Tech mirror:** `git@github.gatech.edu:asoobrian3/generative-first-principles.git` (optional second remote: `gatech`)
