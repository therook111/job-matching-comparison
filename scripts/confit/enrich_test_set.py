"""
Enrich the test dataset with LLM-generated ideal candidate profile summaries.

Mirrors the enrichment step (step 7) of create_confit_dataset.py but operates
on the test split.  Reads from / writes to paths defined in the
``test_enrichment`` block of config.confit.yaml.

Features
--------
- Resume-safe: previously completed enrichments are loaded and skipped.
- Retry mode: if a ``failed_jds`` file exists alongside completed results,
  only the failed JDs are retried.
- Async batch processing via AsyncBatchProcessor (same as training script).
- Writes a side-car JSONL of enriched JDs for easy inspection / re-use.
"""

import csv
import os
import sys
import json
import asyncio

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.utils.async_batch_processor import AsyncBatchProcessor
from src.data_generation.cv_generator import CVGenerator

logger = get_logger(__name__)


def main() -> None:
    # ------------------------------------------------------------------ #
    # 1. Load config                                                       #
    # ------------------------------------------------------------------ #
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    config_path = os.path.join(base_dir, "resources", "config.confit.yaml")
    cfg = ConfigLoader(config_path=config_path)

    test_cfg = cfg.config["test_enrichment"]
    summary_cfg = cfg.config["summary_generation"]

    test_data_dir: str     = test_cfg["test_data_dir"]
    output_path: str       = test_cfg["output_path"]
    summary_out_path: str  = test_cfg["summary_output_dir"]
    failed_jds_path: str   = test_cfg["failed_jds_dir"]
    model_name: str        = summary_cfg["model_name"]

    # Resolve relative paths against the project root.
    for attr in ("test_data_dir", "output_path", "summary_out_path", "failed_jds_path"):
        pass  # resolved inline below

    if not os.path.isabs(test_data_dir):
        test_data_dir = os.path.join(base_dir, test_data_dir)
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)
    if not os.path.isabs(summary_out_path):
        summary_out_path = os.path.join(base_dir, summary_out_path)
    if not os.path.isabs(failed_jds_path):
        failed_jds_path = os.path.join(base_dir, failed_jds_path)

    # ------------------------------------------------------------------ #
    # 2. Load the test dataset                                             #
    # ------------------------------------------------------------------ #
    logger.info(f"Loading test dataset from: {test_data_dir}")
    df = pd.read_csv(test_data_dir)

    required_cols = {"jd", "cv", "match", "classification"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Test dataset is missing required columns: {missing}")

    logger.info(
        f"Loaded {len(df)} rows.  Classifications:\n"
        f"{df['classification'].value_counts().to_dict()}"
    )

    unique_jds: list[str] = df["jd"].unique().tolist()
    logger.info(f"Found {len(unique_jds)} unique JDs to enrich.")

    # ------------------------------------------------------------------ #
    # 3. Resume: load previously completed enrichments                    #
    # ------------------------------------------------------------------ #
    os.makedirs(os.path.dirname(summary_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(failed_jds_path), exist_ok=True)

    completed: dict[str, str] = {}   # {original_jd: enriched_jd}
    if os.path.exists(summary_out_path):
        with open(summary_out_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                completed[record["jd"]] = record["enriched_jd"]
        logger.info(
            f"Loaded {len(completed)} previously completed enrichments "
            f"from: {summary_out_path}"
        )

    # ------------------------------------------------------------------ #
    # 4. Determine which JDs still need processing                        #
    # ------------------------------------------------------------------ #
    if os.path.exists(failed_jds_path) and len(completed) > 0:
        # Retry mode: only process previously failed JDs.
        with open(failed_jds_path, "r", encoding="utf-8") as f:
            jds_to_process = [json.loads(line)["jd"] for line in f]
        logger.info(
            f"Retry mode: processing {len(jds_to_process)} "
            f"previously failed JDs ..."
        )
    else:
        completed_set = set(completed.keys())
        jds_to_process = [jd for jd in unique_jds if jd not in completed_set]
        logger.info(
            f"Processing {len(jds_to_process)} JDs "
            f"({len(completed)} already done, {len(unique_jds)} total) ..."
        )

    # ------------------------------------------------------------------ #
    # 5. Generate profiles via LLM                                        #
    # ------------------------------------------------------------------ #
    cfg.config["generation_model"] = {"model_name": model_name}
    cv_generator = CVGenerator(config_loader=cfg)

    failed_jds: list[str] = []

    def _generate_profile(jd_text: str) -> tuple[str, str]:
        profile = cv_generator.generate_profile(jd_text)
        enriched = f"{jd_text}\n\n## Ideal Candidate Profile\n{profile.summary}"
        return (jd_text, enriched)

    def _on_failure(jd_text: str, exc: Exception) -> None:
        logger.warning(f"Profile generation failed after all retries. Error: {exc}")
        failed_jds.append(jd_text)

    new_results: list[tuple[str, str]] = []
    if jds_to_process:
        processor = AsyncBatchProcessor(
            max_concurrent=10, max_retries=3, initial_backoff=2.0
        )
        raw_results: list[tuple[str, str] | None] = asyncio.run(
            processor.run(
                items=jds_to_process,
                process_fn=_generate_profile,
                on_failure=_on_failure,
                desc="Generating test profiles",
            )
        )
        new_results = [r for r in raw_results if r is not None]
    else:
        logger.info("Nothing new to process.")

    # ------------------------------------------------------------------ #
    # 6. Persist enrichment results                                       #
    # ------------------------------------------------------------------ #
    for original_jd, enriched_jd in new_results:
        completed[original_jd] = enriched_jd

    with open(summary_out_path, "w", encoding="utf-8") as f_out:
        for original_jd, enriched_jd in completed.items():
            f_out.write(
                json.dumps(
                    {"jd": original_jd, "enriched_jd": enriched_jd},
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info(f"Wrote {len(completed)} enriched JDs to: {summary_out_path}")

    if failed_jds:
        with open(failed_jds_path, "w", encoding="utf-8") as f_fail:
            for jd in failed_jds:
                f_fail.write(json.dumps({"jd": jd}, ensure_ascii=False) + "\n")
        logger.warning(
            f"{len(failed_jds)} JDs failed enrichment. Written to: {failed_jds_path}"
        )
    else:
        if os.path.exists(failed_jds_path):
            os.remove(failed_jds_path)
        logger.info("All test JDs enriched successfully.")

    # ------------------------------------------------------------------ #
    # 7. Apply enrichment mapping and save enriched test dataset          #
    # ------------------------------------------------------------------ #
    if len(completed) == len(unique_jds):
        df["jd"] = df["jd"].map(completed)
        logger.info("JD enrichment applied to test dataset.")
    else:
        logger.warning(
            f"Enrichment incomplete ({len(completed)}/{len(unique_jds)} done). "
            f"Skipping JD mapping — fix failures in {failed_jds_path} and re-run."
        )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
    logger.info(f"Saved enriched test dataset ({len(df)} rows) to: {output_path}")
    logger.info(f"Final classifications:\n{df['classification'].value_counts()}")


if __name__ == "__main__":
    main()
