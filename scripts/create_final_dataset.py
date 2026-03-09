
import pandas as pd
import numpy as np
import yaml
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_training_dataset(df, hard_neg_columns, num_easy_negs=3, random_state=42):
    results = []
    
    np.random.seed(random_state)
    all_indices = df.index.tolist()
    cluster_indices = df.groupby('cluster').indices
    total = len(all_indices)
    
    for idx, row in df.iterrows():
        results.append({
            'jd': row['original_jd'],
            'cv': row['positive'],
            'match': 1,
            'classification': 'positive'
        })
        
        for col in hard_neg_columns:
            if pd.notna(row[col]):
                results.append({
                    'jd': row['original_jd'],
                    'cv': row[col],
                    'match': 0,
                    'classification': 'hard_neg'
                })
        
        current_cluster = row['cluster']
        selected_easy_cvs = []
        attempts = 0
        
        while len(selected_easy_cvs) < num_easy_negs:
            random_pos = np.random.randint(0, total)
            random_idx = all_indices[random_pos]
            candidate_cluster = df.at[random_idx, 'cluster']
            
            if candidate_cluster != current_cluster:
                selected_easy_cvs.append(df.at[random_idx, 'positive'])
            
            attempts += 1
            if attempts > num_easy_negs * 50:
                break
        
        for easy_cv in selected_easy_cvs:
            results.append({
                'jd': row['original_jd'],
                'cv': easy_cv,
                'match': 0,
                'classification': 'easy_neg'
            })

    return pd.DataFrame(results)

def main():
    parser = argparse.ArgumentParser(description="Create final training/test dataset.")
    parser.add_argument(
        "--split", type=str, default="train", choices=["train", "test"],
        help="Which data split to process (default: train)"
    )
    args = parser.parse_args()
    split = args.split

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'resources', 'config.yaml')
    config = load_config(config_path)
    
    dataset_config = config['dataset_generation']
    split_config = dataset_config[split]
    
    pos_path = split_config['positive_output_path']
    neg_path = split_config['negative_output_path']
    k_easy_negs = split_config.get('k_easy_negs', 3)
    k_hard_negs = split_config.get('k_hard_negs', 1)
    
    if not os.path.isabs(pos_path):
        pos_path = os.path.join(base_dir, pos_path)
    if not os.path.isabs(neg_path):
        neg_path = os.path.join(base_dir, neg_path)
        
    logger.info(f"Creating dataset for '{split}' split")
    logger.info(f"Loading data from:\n  - {pos_path}\n  - {neg_path}")
    
    try:
        df_pos = pd.read_json(pos_path, lines=True)
        df_neg = pd.read_json(neg_path, lines=True)
    except FileNotFoundError as e:
        logger.error(f"Error loading data: {e}")
        return

    clustered_path = os.path.join(base_dir, "scripts_output", "djinni_jobs_clustered.csv")
    if not os.path.exists(clustered_path):
         logger.error(f"Error: Clustered data file not found at {clustered_path}")
         return

    df_clustered = pd.read_csv(clustered_path)
    df_clustered.rename({'Long Description': 'original_jd'}, axis=1, inplace=True)

    # Pivot hard negatives: each hard_negative_index becomes its own column
    if 'hard_negative_index' in df_neg.columns:
        df_neg_pivot = df_neg.pivot_table(
            index='original_jd',
            columns='hard_negative_index',
            values='hard_negative',
            aggfunc='first'
        ).reset_index()
        hard_neg_columns = [col for col in df_neg_pivot.columns if col != 'original_jd']
        df_neg_pivot.columns = ['original_jd'] + [f'hard_negative_{int(c)}' for c in hard_neg_columns]
        hard_neg_columns = [f'hard_negative_{int(c)}' for c in hard_neg_columns]
    else:
        # Legacy format: single hard_negative column
        df_neg_pivot = df_neg[['original_jd', 'hard_negative']].drop_duplicates(subset='original_jd')
        hard_neg_columns = ['hard_negative']

    logger.info(f"Found {len(hard_neg_columns)} hard negative column(s): {hard_neg_columns}")
    
    logger.info("Merging datasets...")
    merged = df_pos.merge(df_clustered, on='original_jd', how='left')
    final = merged.merge(df_neg_pivot, how='left', on='original_jd')
    
    keep_cols = ['original_jd', 'positive', 'cluster'] + hard_neg_columns
    final = final[keep_cols]
    
    # Drop rows where positive or cluster is missing
    final = final.dropna(subset=['positive', 'cluster'])
    # Drop rows where ALL hard negatives are missing
    final = final.dropna(subset=hard_neg_columns, how='all')
    
    logger.info(f"Processing {len(final)} records with {len(hard_neg_columns)} hard neg(s) and {k_easy_negs} easy neg(s) per JD...")
    
    final_training = create_training_dataset(final, hard_neg_columns=hard_neg_columns, num_easy_negs=k_easy_negs)
    shuffled_df = final_training.sample(frac=1).reset_index(drop=True)
    
    # Save output
    output_path = split_config.get('final_dataset_path')
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)
        
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    shuffled_df.to_csv(output_path, index=False)
    
    logger.info(f"Successfully created {split} dataset with {len(shuffled_df)} rows.")
    logger.info(f"Saved to: {output_path}")
    
    if len(shuffled_df) > 0:
        logger.info(f"Preview:\n{shuffled_df.head()}")

if __name__ == "__main__":
    main()
