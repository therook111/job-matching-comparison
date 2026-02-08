
import pandas as pd
import numpy as np
import yaml
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_training_dataset(df, num_easy_negs=3, random_state=42):
    results = []
    
    np.random.seed(random_state)
    cluster_indices = df.groupby('cluster').indices
    total = len(df)
    
    for idx, row in df.iterrows():
        results.append({
            'jd': row['original_jd'],
            'cv': row['positive'],
            'match': 1,
            'classification': 'positive'
        })
        
        results.append({
            'jd': row['original_jd'],
            'cv': row['hard_negative'],
            'match': 0,
            'classification': 'hard_neg'
        })
        
        current_cluster = row['cluster']
        selected_easy_cvs = []
        attempts = 0
        
        while len(selected_easy_cvs) < num_easy_negs:
            random_idx = np.random.randint(0, total)
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
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = os.path.join(base_dir, 'resources', 'config.yaml')
    config = load_config(config_path)
    
    pos_path = config['dataset_generation']['positive_output_path']
    neg_path = config['dataset_generation']['negative_output_path']
    k_easy_negs = config['dataset_generation'].get('k_easy_negs', 3)
    
    if not os.path.isabs(pos_path):
        pos_path = os.path.join(base_dir, pos_path)
    if not os.path.isabs(neg_path):
        neg_path = os.path.join(base_dir, neg_path)
        
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
    
    logger.info("Merging datasets...")
    merged = df_pos.merge(df_clustered, on='original_jd', how='left')
    final = merged.merge(df_neg, how='left', on='original_jd')
    final = final[['original_jd', 'positive', 'hard_negative', 'cluster']]
    final = final.dropna(subset=['positive', 'hard_negative', 'cluster'])
    
    logger.info(f"Processing {len(final)} records with {k_easy_negs} easy negatives per JD...")
    
    final_training = create_training_dataset(final, num_easy_negs=k_easy_negs)
    shuffled_df = final_training.sample(frac=1).reset_index(drop=True)
    
    # Save output
    output_path = config['dataset_generation'].get('final_training_dataset_path')
    if not os.path.isabs(output_path):
        output_path = os.path.join(base_dir, output_path)
        
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    shuffled_df.to_csv(output_path, index=False)
    
    logger.info(f"Successfully created training dataset with {len(shuffled_df)} rows.")
    logger.info(f"Saved to: {output_path}")
    
    if len(shuffled_df) > 0:
        logger.info(f"Preview:\n{shuffled_df.head()}")

if __name__ == "__main__":
    main()
