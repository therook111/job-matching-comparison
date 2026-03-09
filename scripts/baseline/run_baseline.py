import os
import json 
import pandas as pd
import argparse

from src.utils.config_loader import ConfigLoader
from src.methods.baseline import (
    assign_cosine_score, 
    calculate_metrics_extended
)
from src.utils.logger import get_logger 

logger = get_logger(__name__)

config_loader = ConfigLoader('resources/config.baseline.yaml')

embedding_model = config_loader.load_sentence_transformer()

def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluation on a dataset.")
    parser.add_argument(
        "--split", type=str, default="test", choices=["test", "qa"],
        help="Which data split to process (default: test)"
    )

    args = parser.parse_args()
    split = args.split

    logger.info("Running baseline evaluation on split: %s", split)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    split_config = config_loader.get("evaluation_datasets")[split]
    data_path = split_config["dataset_dir"]
    output_path = split_config["output_path"]

    df = pd.read_csv(os.path.join(base_dir, data_path))
    logger.info("Loaded dataset with %d rows", len(df))

    logger.info("Running cosine score assignment")
    df = assign_cosine_score(df, embedding_model)

    logger.info("Running metric calculation")
    results = df.groupby('jd').apply(calculate_metrics_extended)
    final_metrics = results.mean()

    logger.info("Final metrics: %s", final_metrics)

    with open(os.path.join(base_dir, output_path), "w") as f:
        json.dump(final_metrics.to_dict(), f)

if __name__ == "__main__":
    main()






