import os
import sys
import json
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np
import pandas as pd
import torch
import optuna
from pathlib import Path
from sentence_transformers import SentenceTransformer
from torch_geometric.loader import DataLoader

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.methods.GNN.schemas import ExtractedEntity
from src.methods.GNN.training.dataset import CJMDataset
from src.methods.GNN.training.model import CJMGCN
from src.methods.GNN.training.loss import binary_loss
from src.methods.GNN.training.constants import PARAM_GRID

logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
GNN_CONFIG_PATH = BASE_DIR / "resources" / "config.gnn.yaml"


def load_config():
    cfg = ConfigLoader(str(GNN_CONFIG_PATH))

    def abs_path(p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else BASE_DIR / p

    data_cfg  = cfg.config["data"]
    train_cfg = cfg.config["training"]
    embed_cfg = cfg.config["embedding_model"]
    knn_cfg   = cfg.config["common_neighbors_similarity"]

    return {
        "train_csv":        abs_path(data_cfg["train_dataset_path"]),
        "test_csv":         abs_path(data_cfg["test_dataset_path"]),
        "jd_jsonl":         abs_path(data_cfg["extracted_jd_path"]),
        "cv_jsonl":         abs_path(data_cfg["extracted_cv_path"]),
        "train_ratio":      data_cfg["train_ratio"],
        "val_ratio":        data_cfg["val_ratio"],
        "trials":           train_cfg["trials_per_config"],
        "max_epochs":       train_cfg["max_epochs"],
        "patience":         train_cfg["patience"],
        "weight_decay":     train_cfg["weight_decay"],
        "embed_model_name": embed_cfg["name"],
        "embed_device":     embed_cfg["device"],
        "k_neighbors":      knn_cfg["k"],
        "p_sharpening":     knn_cfg["sharpening_factor"],
    }


def load_jsonl(path: Path) -> dict[str, ExtractedEntity]:
    entities: dict[str, ExtractedEntity] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            text = record.pop("text")
            entities[text] = ExtractedEntity(**record)
    return entities


def build_data_list(
    csv_path: Path,
    jd_map: dict[str, ExtractedEntity],
    cv_map: dict[str, ExtractedEntity],
) -> list[tuple[ExtractedEntity, ExtractedEntity, int]]:
    df = pd.read_csv(csv_path)
    data_list, skipped = [], 0
    for _, row in df.iterrows():
        jd = jd_map.get(row["jd"])
        cv = cv_map.get(row["cv"])
        if jd is None or cv is None:
            skipped += 1
            continue
        data_list.append((cv, jd, int(row["match"])))
    if skipped:
        logger.warning(f"Skipped {skipped} rows missing from JSONL maps.")
    return data_list


def split_data(
    data_list: list,
    train_ratio: float,
    val_ratio: float,
    seed: int = 42,
) -> tuple[list, list, list]:
    random.seed(seed)
    shuffled = data_list[:]
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)
    return shuffled[:n_train], shuffled[n_train:n_train + n_val], shuffled[n_train + n_val:]


def run_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    with torch.set_grad_enabled(train):
        for batch in loader:
            batch = batch.to(device)
            logits = model(
                batch.x_primary, batch.x_entities,
                batch.edge_index, batch.edge_weight, batch.batch,
            )
            loss = binary_loss(logits, batch.y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def train_model(params: dict, train_ds, val_ds, cfg: dict, device: torch.device) -> float:
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64, shuffle=False)

    model = CJMGCN(
        hidden_channels=params["hidden_channels"],
        num_gc_layers=params["num_gc_layers"],
        dropout=params["dropout"],
        use_jumping_knowledge=params["use_jumping_knowledge"],
        num_readout_layers=params["num_readout_layers"],
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=params["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(cfg["max_epochs"]):
        run_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, optimizer, device, train=False)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                logger.info(f"Early stopping at epoch {epoch + 1}.")
                break

    return best_val_loss


def make_objective(train_ds, val_ds, cfg, device):
    def objective(trial: optuna.Trial) -> float:
        params = {
            key: trial.suggest_categorical(key, values)
            for key, values in PARAM_GRID.items()
        }
        return train_model(params, train_ds, val_ds, cfg, device)
    return objective


def main():
    cfg = load_config()
    device = torch.device(cfg["embed_device"] if torch.cuda.is_available() else "cpu")

    logger.info(f"Loading embedding model: {cfg['embed_model_name']}")
    st_model = SentenceTransformer(cfg["embed_model_name"], device=str(device))

    def embedder(text: str) -> torch.Tensor:
        return torch.tensor(st_model.encode(text, normalize_embeddings=True))

    logger.info("Loading extracted entities ...")
    jd_map = load_jsonl(cfg["jd_jsonl"])
    cv_map = load_jsonl(cfg["cv_jsonl"])
    logger.info(f"JDs: {len(jd_map)}  |  CVs: {len(cv_map)}")

    data_list = build_data_list(cfg["train_csv"], jd_map, cv_map)
    logger.info(f"Total valid pairs: {len(data_list)}")

    train_pairs, val_pairs, test_pairs = split_data(data_list, cfg["train_ratio"], cfg["val_ratio"])
    logger.info(f"Split — train: {len(train_pairs)}, val: {len(val_pairs)}, test: {len(test_pairs)}")

    def make_dataset(pairs):
        return CJMDataset(
            data_list=pairs,
            embedder=embedder,
            k_neighbors=cfg["k_neighbors"],
            p_sharpening=cfg["p_sharpening"],
        )

    train_ds = make_dataset(train_pairs)
    val_ds   = make_dataset(val_pairs)
    test_ds  = make_dataset(test_pairs)

    logger.info(f"Starting Optuna study — {cfg['trials']} trials ...")
    study = optuna.create_study(direction="minimize")
    study.optimize(make_objective(train_ds, val_ds, cfg, device), n_trials=cfg["trials"])

    logger.info(f"Best trial val_loss: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")

    logger.info("Retraining best model on train + val ...")
    full_train_ds = make_dataset(train_pairs + val_pairs)
    best_model_loss = train_model(study.best_params, full_train_ds, test_ds, cfg, device)
    logger.info(f"Test loss with best params: {best_model_loss:.4f}")


if __name__ == "__main__":
    main()
