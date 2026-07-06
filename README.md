# Generative Modeling from First Principles

Interactive notebook on VAEs, GANs, and diffusion — math derived, then verified in code.

## Students: run in Google Colab

1. Clone or download `generative_first_principles.ipynb` from this repository.
2. Open [Google Colab](https://colab.research.google.com/) → **File → Upload notebook**.
3. **Runtime → Change runtime type → GPU** (recommended; CPU works with `QUICK=True`).
4. **Runtime → Run all**.

On a fresh Colab runtime, if imports fail, run once:

```python
!pip install -q scikit-learn scipy
```

### Settings in the notebook

| Variable | Default | Effect |
|----------|---------|--------|
| `QUICK` | `False` | `True` → 8×8 + fast training; `False` → 32×32 (readable digits) |
| `IMG` | from `QUICK` | `8` if `QUICK` else `32` |
| `SEED` | `0` | Reproducible runs |

Use `QUICK=True` for a quick smoke test; `QUICK=False` for assignment-quality figures.

## Local setup (optional)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab generative_first_principles.ipynb
```

## Files

| File | Purpose |
|------|---------|
| `generative_first_principles.ipynb` | Main notebook (distribute this) |
| `vae.ipynb` | VAE-focused extract (work in progress) |
| `build_notebook.py` | Regenerates the notebook from source (instructors) |
| `requirements.txt` | Python dependencies |
