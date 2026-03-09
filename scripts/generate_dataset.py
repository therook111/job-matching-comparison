import pandas as pd
import os
import json
import sys
import asyncio
import random
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data_generation.dataset_generator import DatasetGenerator
from src.utils.logger import get_logger
from src.utils.async_batch_processor import AsyncBatchProcessor
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)


def load_processed_jds(file_path):
    processed = set()
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "original_jd" in data:
                        processed.add(data["original_jd"])
                except:
                    pass
    return processed


async def generate_data_async(split: str = "train"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return

    input_path = os.path.join("scripts_output", f"djinni_{split}_seed_jd.csv")
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return
    
    df = pd.read_csv(input_path)
    
    logger.info(f"Initializing DatasetGenerator for '{split}' split...")
    generator = DatasetGenerator(api_key=api_key, split=split)

    processed_positive = load_processed_jds(generator.positive_output_path)
    processed_negative = load_processed_jds(generator.negative_output_path)
    processed_jds = processed_positive.union(processed_negative)
    logger.info(f"Found {len(processed_jds)} already processed JDs.")

    jd_texts = [
        row["Long Description"]
        for _, row in df.iterrows()
        if isinstance(row["Long Description"], str) and row["Long Description"].strip()
    ]

    initial_count = len(jd_texts)
    jd_texts = [text for text in jd_texts if text not in processed_jds]
    logger.info(f"Skipping {initial_count - len(jd_texts)} JDs. Processing {len(jd_texts)} new JDs.")

    failed_output_path = os.path.join("scripts_output", f"failed_jds_{split}.jsonl")
    if os.path.exists(failed_output_path):
        os.remove(failed_output_path)

    # Prepare items with language
    items = [
        (jd_text, "Vietnamese" if random.random() < 0.5 else "English")
        for jd_text in jd_texts
    ]

    def process_fn(item):
        jd_text, language = item
        generator.generate_dataset_entry(jd_text, language)
        return True

    def on_failure(item, exception):
        jd_text, _ = item
        error_data = json.dumps({
            "original_jd": jd_text,
            "error": str(exception)
        }, ensure_ascii=False) + '\n'
        with open(failed_output_path, 'a', encoding='utf-8') as f:
            f.write(error_data)

    processor = AsyncBatchProcessor(max_concurrent=36, max_retries=3, initial_backoff=2.0)
    results = await processor.run(items, process_fn, on_failure, desc=f"Generating CVs ({split})")

    success_count = sum(1 for r in results if r is not None)
    failure_count = len(results) - success_count

    logger.info("Generation complete.")
    logger.info(f"Successfully generated entries: {success_count}")
    logger.info(f"Failed entries: {failure_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic CV dataset.")
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "test"],
        help="Which data split to generate for (default: train)"
    )
    args = parser.parse_args()
    asyncio.run(generate_data_async(split=args.split))