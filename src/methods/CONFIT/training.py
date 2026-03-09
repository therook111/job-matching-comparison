"""
Adaptation of ConFitv2:  Improving Resume-Job Matching using Hypothetical Resume
Embedding and Runner-Up Hard-Negative Mining by Xiao Yu et al. (2025).

This implementation includes:
- JD->CV direction loss calculation only (due to the nature of the generative dataset)
- Offline Runner-Up Hard-Negative Mining
"""

import os
import sys
import pandas as pd
from pathlib import Path

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments
from sentence_transformers.losses import MultipleNegativesRankingLoss

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.methods.confit.utils import convert_to_hf_dataset

logger = get_logger(__name__)


def load_config() -> ConfigLoader:
    """Load the ConFit config from config.confit.yaml."""
    base_dir = Path(__file__).resolve().parents[3]
    config_path = base_dir / "resources" / "config.confit.yaml"
    return ConfigLoader(config_path=str(config_path))

def train(cfg: ConfigLoader | None = None) -> None:
    """
    Train a bi-encoder model using MultipleNegativesRankingLoss.

    Parameters
    ----------
    cfg : ConfigLoader, optional
        Pre-loaded config. If None the config is loaded from disk.
    """
    if cfg is None:
        cfg = load_config()

    confit_cfg   = cfg.config["confit"]
    training_cfg = confit_cfg["training"]

    # ── Paths ──────────────────────────────────────────────────────────────
    base_dir     = Path(__file__).resolve().parents[3]
    dataset_path = cfg.config["rum"]["output_path"]
    output_dir   = training_cfg["output_dir"]

    if not os.path.isabs(dataset_path):
        dataset_path = str(base_dir / dataset_path)
    if not os.path.isabs(output_dir):
        output_dir = str(base_dir / output_dir)

    logger.info(f"ConFit config: {confit_cfg}")
    logger.info(f"Training config: {training_cfg}")

    # ── 1. Load dataset ────────────────────────────────────────────────────
    logger.info(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    logger.info(f"Loaded {len(df)} rows.")

    hf_dataset = convert_to_hf_dataset(df)
    logger.info(f"Grouped into {len(hf_dataset)} unique JD triplets.")

    # Train/eval split
    split = hf_dataset.train_test_split(test_size=1 - confit_cfg["split_ratio"], seed=confit_cfg["seed"])
    train_dataset = split["train"]
    eval_dataset  = split["test"]
    logger.info(f"Train triplets: {len(train_dataset)}  |  Eval triplets: {len(eval_dataset)}")

    # ── 2. Load model ──────────────────────────────────────────────────────
    logger.info(f"Loading SentenceTransformer: {confit_cfg['base_encoder']}")
    model = SentenceTransformer(confit_cfg["base_encoder"])
    model.max_seq_length = confit_cfg["max_seq_length"]

    # ── 3. Loss ────────────────────────────────────────────────────────────
    # MultipleNegativesRankingLoss treats (anchor, positive, *negatives) triplets.
    # In-batch negatives are automatically generated from other pairs, so explicit
    # negatives in the `negative` column are used as extra hard negatives.
    loss = MultipleNegativesRankingLoss(model=model)

    # ── 4. Training arguments ──────────────────────────────────────────────
    training_args = SentenceTransformerTrainingArguments(
        output_dir=output_dir,
        num_train_epochs=training_cfg["epochs"],
        seed=confit_cfg["seed"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_steps=50,
        **{k: training_cfg[k] for k in (
            "per_device_train_batch_size",
            "per_device_eval_batch_size",
            "learning_rate",
            "warmup_ratio",
            "gradient_accumulation_steps",
            "fp16",
        )},
    )

    # ── 5. Trainer ─────────────────────────────────────────────────────────
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
    )

    logger.info("Starting training ...")
    trainer.train()
    logger.info("Training complete.")

    # ── 6. Save final model ────────────────────────────────────────────────
    final_model_path = os.path.join(output_dir, "final")
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")

