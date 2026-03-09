import os
import sys
import json
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from src.utils.config_loader import ConfigLoader
from src.utils.logger import get_logger
from src.utils.async_batch_processor import AsyncBatchProcessor
from src.data_generation.parser import DataParser
from src.methods.GNN.schemas import ExtractedEntity
from src.methods.GNN.prompts import PARSER_PROMPT

load_dotenv()
logger = get_logger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
GNN_CONFIG_PATH = BASE_DIR / "resources" / "config.gnn.yaml"


def load_config():
    cfg = ConfigLoader(str(GNN_CONFIG_PATH))
    data_cfg = cfg.config["data"]

    def abs_path(p: str) -> Path:
        p = Path(p)
        return p if p.is_absolute() else BASE_DIR / p

    return {
        "train_path": abs_path(data_cfg["train_dataset_path"]),
        "test_path":  abs_path(data_cfg["test_dataset_path"]),
        "jd_out":     abs_path(data_cfg["extracted_jd_path"]),
        "cv_out":     abs_path(data_cfg["extracted_cv_path"]),
    }, cfg


def load_completed(path: Path) -> dict[str, dict]:
    completed: dict[str, dict] = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    completed[record["text"]] = record
                except json.JSONDecodeError:
                    pass
    return completed


def append_record(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def extract_texts(texts: list[str], out_path: Path, parser: DataParser, label: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    completed = load_completed(out_path)
    to_process = [t for t in texts if t not in completed]

    logger.info(f"[{label}] {len(completed)} done, {len(to_process)} remaining / {len(texts)} unique.")

    if not to_process:
        return

    failed_path = out_path.with_suffix(".failed.jsonl")

    def process_fn(text: str) -> dict:
        prompt = PARSER_PROMPT.format(document=text)
        entity: ExtractedEntity = parser.parse(text=text, schema=ExtractedEntity, prompt_template=prompt)
        return {"text": text, **entity.model_dump()}

    def on_failure(text: str, exc: Exception) -> None:
        logger.warning(f"[{label}] Failed: {exc}")
        with open(failed_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"text": text, "error": str(exc)}, ensure_ascii=False) + "\n")

    processor = AsyncBatchProcessor(max_concurrent=10, max_retries=3, initial_backoff=2.0)
    results: list[dict | None] = asyncio.run(
        processor.run(items=to_process, process_fn=process_fn, on_failure=on_failure, desc=f"Extracting {label}")
    )

    success = sum(1 for r in results if r is not None)
    for record in results:
        if record is not None:
            append_record(out_path, record)

    logger.info(f"[{label}] {success} succeeded, {len(to_process) - success} failed.")
    if failed_path.exists():
        logger.warning(f"[{label}] Failed entries → {failed_path}")


def main():
    paths, cfg = load_config()

    dfs = []
    for split_path in (paths["train_path"], paths["test_path"]):
        if not split_path.exists():
            logger.warning(f"Not found, skipping: {split_path}")
            continue
        logger.info(f"Loading: {split_path}")
        dfs.append(pd.read_csv(split_path))

    if not dfs:
        logger.error("No dataset files found. Aborting.")
        return

    df = pd.concat(dfs, ignore_index=True)
    unique_jds = df["jd"].dropna().unique().tolist()
    unique_cvs = df["cv"].dropna().unique().tolist()
    logger.info(f"Unique JDs: {len(unique_jds)}  |  Unique CVs: {len(unique_cvs)}")

    parser = DataParser()

    extract_texts(texts=unique_jds, out_path=paths["jd_out"], parser=parser, label="JD")
    extract_texts(texts=unique_cvs, out_path=paths["cv_out"], parser=parser, label="CV")

    logger.info(f"Done.  JDs → {paths['jd_out']}  |  CVs → {paths['cv_out']}")


if __name__ == "__main__":
    main()
