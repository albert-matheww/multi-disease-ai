"""MultiDiseaseAI end-to-end pipeline orchestrator.

Runs (a subset of) download -> preprocess -> train -> evaluate -> explain
for one or all four diseases. Each stage is idempotent and can also be run
standalone via its own module (see each stage's --help).

Examples:
    python main.py                          # full pipeline, all diseases
    python main.py --disease heart          # full pipeline, heart only
    python main.py --stages train,evaluate  # only (re)train + (re)evaluate
    python main.py --skip-download          # reuse already-downloaded CSVs
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

STAGES = ["download", "preprocess", "train", "evaluate", "explain"]


def run_download() -> None:
    from scripts.download_data import main as download_main

    download_main()


def run_preprocess(disease_keys: list[str]) -> None:
    from src.config import get_disease
    from src.preprocessing import DiseasePreprocessor

    for key in disease_keys:
        DiseasePreprocessor(get_disease(key)).run()


def run_train(disease_keys: list[str], device: str) -> None:
    from src.train import train_disease

    for key in disease_keys:
        train_disease(key, device=device)


def run_evaluate(disease_keys: list[str]) -> None:
    from src.evaluate import evaluate_disease

    for key in disease_keys:
        evaluate_disease(key)


def run_explain(disease_keys: list[str]) -> None:
    from src.explainability import generate_global_explanations

    for key in disease_keys:
        generate_global_explanations(key)


def main() -> None:
    from src.config import DISEASES

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--disease", choices=list(DISEASES), default=None, help="Run for a single disease only"
    )
    parser.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"Comma-separated subset of stages to run: {STAGES}",
    )
    parser.add_argument(
        "--skip-download", action="store_true", help="Shorthand for --stages without 'download'"
    )
    parser.add_argument(
        "--device", default="cpu", help="TabPFN device for the train stage: cpu, cuda, mps, auto"
    )
    args = parser.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]
    if args.skip_download and "download" in stages:
        stages.remove("download")

    unknown = set(stages) - set(STAGES)
    if unknown:
        parser.error(f"Unknown stage(s): {sorted(unknown)}. Valid stages: {STAGES}")

    disease_keys = [args.disease] if args.disease else list(DISEASES)

    logger.info("Running pipeline stages=%s for diseases=%s", stages, disease_keys)

    if "download" in stages:
        run_download()
    if "preprocess" in stages:
        run_preprocess(disease_keys)
    if "train" in stages:
        run_train(disease_keys, device=args.device)
    if "evaluate" in stages:
        run_evaluate(disease_keys)
    if "explain" in stages:
        run_explain(disease_keys)

    logger.info("Pipeline complete.")


if __name__ == "__main__":
    main()
