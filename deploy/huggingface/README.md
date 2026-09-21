---
title: MultiDiseaseAI
emoji: 🩺
colorFrom: indigo
colorTo: red
sdk: docker
app_port: 8501
pinned: false
license: mit
short_description: Uncertainty-aware multi-disease risk prediction with TabPFN
---

# MultiDiseaseAI — Hugging Face Space

This directory holds everything specific to deploying the project as a
[Hugging Face **Docker** Space](https://huggingface.co/docs/hub/spaces-sdks-docker).
The Space runs the exact same `Dockerfile` and Streamlit app as the repo root;
only this card (`README.md`) and the sync script live here.

## One-time setup

1. **Create the Space** (or let the workflow create it on first run):
   `https://huggingface.co/new-space` → *Docker* SDK → blank template.
2. **Add repository secrets / variables on GitHub** (Settings → Secrets and
   variables → Actions):
   - secret `HF_TOKEN` — a Hugging Face access token with *write* scope.
   - variable `HF_SPACE` — the target, e.g. `your-username/MultiDiseaseAI`.
3. **Bundle trained models** (the "pre-train + bundle" flow). TabPFN needs a
   one-time license token to download its weights, so training happens on your
   machine, not in CI:

   ```bash
   cp .env.example .env            # add TABPFN_TOKEN
   python scripts/download_data.py
   python main.py --stages preprocess,train,evaluate,explain
   git lfs install                 # *_tabpfn.joblib is LFS-tracked (.gitattributes)
   git add models/*.joblib reports/*_metrics.json reports/figures
   git commit -m "Bundle trained TabPFN models + evaluation artifacts"
   git push
   ```

## Deploying

Push to `main` (or run the **Deploy to Hugging Face Space** workflow manually).
The workflow mirrors the repo — including the LFS-tracked model bundles — into
the Space, which then rebuilds the image and restarts.

## Runtime notes

- The app **loads** the bundled `models/*_tabpfn.joblib`; it never trains.
- If a bundled model still triggers a TabPFN weight download on first predict,
  either add `TABPFN_TOKEN` as a **Space secret** (Settings → Variables and
  secrets) so weights download once at runtime, or build the image with
  `--build-arg TABPFN_TOKEN=…` locally to bake weights in (see `Dockerfile`).
- The prediction-history SQLite file lives inside the container and resets when
  the Space restarts — expected for a public demo. Attach a persistent
  `/data` volume and point `DB_PATH` there if you need it to survive restarts.
