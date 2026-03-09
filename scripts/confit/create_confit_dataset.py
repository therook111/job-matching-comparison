"""
This script is used to create the dataset for the ConFit model.

Features:
- Ideal candidate summary anchored to a specific JD
- RUM (Runner-up Mining) hard negatives
"""

import os
import sys
import json
import asyncio
import random
import numpy as np
import pandas as pd
import faiss
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.utils.async_batch_processor import AsyncBatchProcessor
from src.data_generation.cv_generator import CVGenerator

logger = get_logger(__name__)


def main():
    # 1. Load config
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    config_path = os.path.join(base_dir, "resources", "config.confit.yaml")
    cfg = ConfigLoader(config_path=config_path)

    rum_cfg = cfg.config["rum"]
    percentile_lower: float = rum_cfg["percentile_lower"]
    percentile_upper: float = rum_cfg["percentile_upper"]
    batch_size: int         = rum_cfg["batch_size"]
    seed: int               = rum_cfg["seed"]
    k_mined_negs: int       = rum_cfg["k_mined_negs"]
    train_data_dir: str     = rum_cfg["train_data_dir"]
    output_path: str        = rum_cfg["output_path"]

    model_name: str = cfg.config["confit"]["base_encoder"]

    if not os.path.isabs(train_data_dir):
        train_data_dir = os.path.join(base_dir, train_data_dir)
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)

    random.seed(seed)
    np.random.seed(seed)

    if os.path.exists(output_path):
        # Checkpoint: output already exists — skip steps 2-6 and go straight to enrichment.
        logger.info(f"Checkpoint found at {output_path}, skipping steps 2-6.")
        df_augmented = pd.read_csv(output_path)
    else:
        # 2. Load dataset
        logger.info(f"Loading dataset from: {train_data_dir}")
        df = pd.read_csv(train_data_dir)

        required_cols = {"jd", "cv", "match", "classification"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Dataset is missing required columns: {missing}")

        logger.info(f"Loaded {len(df)} rows. Classifications: {df['classification'].value_counts().to_dict()}")

        # 3. Embed unique CVs
        logger.info(f"Loading encoder: {model_name}")
        encoder = SentenceTransformer(model_name)

        unique_cv_texts = list(dict.fromkeys(df["cv"].tolist()))
        logger.info(f"Embedding {len(unique_cv_texts)} unique CVs (from {len(df)} total rows) ...")
        cv_embeddings = encoder.encode(
            unique_cv_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        # 4. Build FAISS index
        dim = cv_embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(cv_embeddings)
        logger.info(f"FAISS index built with {index.ntotal} unique vectors (dim={dim})")

        jd_to_positive_cvs: dict[str, set[str]] = (
            df[df["match"] == 1]
            .groupby("jd")["cv"]
            .apply(set)
            .to_dict()
        )

        # 5. Mine RUM negatives
        total_cvs = len(unique_cv_texts)
        rank_lo = int(np.floor(percentile_lower * total_cvs))
        rank_hi = int(np.ceil(percentile_upper  * total_cvs))
        k_retrieve = rank_hi + 1

        unique_jds = df["jd"].unique()
        logger.info(f"Mining RUM negatives for {len(unique_jds)} unique JDs "
                    f"(percentile window [{percentile_lower}, {percentile_upper}] → "
                    f"rank window [{rank_lo}, {rank_hi}), k_mined_negs={k_mined_negs})")

        jd_texts = unique_jds.tolist()
        logger.info(f"Embedding {len(jd_texts)} unique JDs ...")
        jd_embeddings = encoder.encode(
            jd_texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        new_rows = []

        for jd_text, jd_emb in tqdm(zip(jd_texts, jd_embeddings), total=len(jd_texts), desc="Mining RUM negatives"):
            query = jd_emb[np.newaxis, :]
            _, I = index.search(query, k_retrieve)
            ranked_indices = I[0]

            window_indices = ranked_indices[rank_lo:rank_hi].tolist()

            positive_cvs = jd_to_positive_cvs.get(jd_text, set())
            window_indices = [
                i for i in window_indices
                if i >= 0 and unique_cv_texts[i] not in positive_cvs
            ]

            if len(window_indices) == 0:
                continue

            sampled_indices = (
                random.sample(window_indices, k_mined_negs)
                if len(window_indices) > k_mined_negs
                else window_indices
            )

            for cv_idx in sampled_indices:
                new_rows.append({
                    "jd":             jd_text,
                    "cv":             unique_cv_texts[cv_idx],
                    "match":          0,
                    "classification": "rum_neg",
                })

        logger.info(f"Mined {len(new_rows)} RUM negative pairs.")

        # 6. Append mined negatives, shuffle, save
        df_new = pd.DataFrame(new_rows, columns=["jd", "cv", "match", "classification"])
        df_augmented = pd.concat([df, df_new], ignore_index=True)
        df_augmented = df_augmented.sample(frac=1, random_state=seed).reset_index(drop=True)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_augmented.to_csv(output_path, index=False)
        logger.info(f"Saved {len(df_augmented)} rows (pre-enrichment) to: {output_path}")

    # 7. Enrich every JD with an LLM-generated ideal candidate summary
    summary_cfg          = cfg.config["summary_generation"]
    summary_model_name: str = summary_cfg["model_name"]
    summary_out_path: str   = summary_cfg["output_dir"]
    failed_jds_path: str    = summary_cfg["failed_jds_dir"]

    if not os.path.isabs(summary_out_path):
        summary_out_path = os.path.join(base_dir, summary_out_path)
    if not os.path.isabs(failed_jds_path):
        failed_jds_path = os.path.join(base_dir, failed_jds_path)

    os.makedirs(os.path.dirname(summary_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(failed_jds_path), exist_ok=True)

    unique_jds_for_profile = df_augmented["jd"].unique().tolist()

    # Resume: load any previously completed enrichments.
    completed: dict[str, str] = {}  # {original_jd: enriched_jd}
    if os.path.exists(summary_out_path):
        with open(summary_out_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                completed[record["jd"]] = record["enriched_jd"]
        logger.info(f"Loaded {len(completed)} previously completed enrichments from: {summary_out_path}")

    # Determine which JDs still need processing.
    if os.path.exists(failed_jds_path) and len(completed) > 0:
        # Retry mode: only process the JDs that previously failed.
        with open(failed_jds_path, "r", encoding="utf-8") as f:
            jds_to_process = [json.loads(line)["jd"] for line in f]
        logger.info(f"Retry mode: processing {len(jds_to_process)} previously failed JDs ...")
    else:
        # Fresh / first-run: process everything not yet completed.
        completed_set = set(completed.keys())
        jds_to_process = [jd for jd in unique_jds_for_profile if jd not in completed_set]
        logger.info(f"Processing {len(jds_to_process)} JDs ({len(completed)} already done, {len(unique_jds_for_profile)} total) ...")

    cfg.config["generation_model"] = {"model_name": summary_model_name}
    cv_generator = CVGenerator(config_loader=cfg)

    failed_jds: list[str] = []

    def _generate_profile(jd_text: str) -> tuple[str, str]:
        profile = cv_generator.generate_profile(jd_text)
        return (jd_text, f"{jd_text}\n\n## Ideal Candidate Profile\n{profile.summary}")

    def _on_failure(jd_text: str, exc: Exception) -> None:
        logger.warning(f"Profile generation failed after all retries. Error: {exc}")
        failed_jds.append(jd_text)

    new_results: list[tuple[str, str]] = []
    if jds_to_process:
        processor = AsyncBatchProcessor(max_concurrent=10, max_retries=3, initial_backoff=2.0)
        raw_results: list[tuple[str, str] | None] = asyncio.run(
            processor.run(
                items=jds_to_process,
                process_fn=_generate_profile,
                on_failure=_on_failure,
                desc="Generating profiles",
            )
        )
        new_results = [r for r in raw_results if r is not None]
    else:
        logger.info("Nothing new to process.")

    # Merge new successes into the completed dict and write the full summary JSONL.
    for original_jd, enriched_jd in new_results:
        completed[original_jd] = enriched_jd

    with open(summary_out_path, "w", encoding="utf-8") as f_out:
        for original_jd, enriched_jd in completed.items():
            f_out.write(json.dumps({"jd": original_jd, "enriched_jd": enriched_jd}, ensure_ascii=False) + "\n")
    logger.info(f"Wrote {len(completed)} enriched JDs to: {summary_out_path}")

    if failed_jds:
        with open(failed_jds_path, "w", encoding="utf-8") as f_fail:
            for jd in failed_jds:
                f_fail.write(json.dumps({"jd": jd}, ensure_ascii=False) + "\n")
        logger.warning(f"{len(failed_jds)} JDs failed enrichment. Written to: {failed_jds_path}")
    else:
        # No failures this run — remove a stale failed file if present.
        if os.path.exists(failed_jds_path):
            os.remove(failed_jds_path)
        logger.info("All JDs enriched successfully.")

    # Only apply the mapping if every unique JD is now accounted for.
    if len(completed) == len(unique_jds_for_profile):
        df_augmented["jd"] = df_augmented["jd"].map(completed)
        logger.info("JD enrichment applied to dataset.")
    else:
        logger.warning(
            f"Enrichment incomplete ({len(completed)}/{len(unique_jds_for_profile)} done). "
            f"Skipping JD mapping — fix failures in {failed_jds_path} and re-run."
        )





    # 8. Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_augmented.to_csv(output_path, index=False)

    logger.info(f"Saved augmented dataset ({len(df_augmented)} rows) to: {output_path}")
    logger.info(f"Final classifications:\n{df_augmented['classification'].value_counts()}")


if __name__ == "__main__":
    main()
