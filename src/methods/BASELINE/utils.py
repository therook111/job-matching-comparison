import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import paired_cosine_distances
from sentence_transformers import SentenceTransformer

def assign_cosine_score(df, model: SentenceTransformer):
    jd_embeddings = model.encode(df['jd'].tolist(), convert_to_numpy=True, show_progress_bar=True)
    cv_embeddings = model.encode(df['cv'].tolist(), convert_to_numpy=True, show_progress_bar=True)
    
    cosine_scores = 1 - paired_cosine_distances(jd_embeddings, cv_embeddings)

    # Assign scores back to dataframe
    df['score'] = cosine_scores

    return df

def calculate_metrics_extended(group):

    sorted_group = group.sort_values(by='score', ascending=False).reset_index(drop=True)
    
    # 2. Identify Rank of Positive
    try:
        pos_idx = sorted_group[sorted_group['classification'] == 'positive'].index[0]
        pos_rank = pos_idx + 1 # 1-based rank
    except IndexError:
        return None 

    # 3. Analyze Hard Negatives (Count how many are ranked ABOVE the positive)
    # We filter for Hard Negs that have an index LOWER than the positive index
    hard_negs_above = sorted_group[
        (sorted_group['classification'] == 'hard_neg') & 
        (sorted_group.index < pos_idx)
    ]
    
    num_hard_negs_winning = len(hard_negs_above)
    
    # Binary Failure: Did AT LEAST ONE hard negative beat the positive?
    hard_neg_failure_binary = 1 if num_hard_negs_winning > 0 else 0

    # 4. Analyze Easy Negatives (Sanity Check)
    easy_negs_above = sorted_group[
        (sorted_group['classification'] == 'easy_neg') & 
        (sorted_group.index < pos_idx)
    ]
    num_easy_negs_winning = len(easy_negs_above)
    easy_neg_failure_binary = 1 if num_easy_negs_winning > 0 else 0

    # 5. Standard Metrics
    k = 5
    ndcg = 0
    recall = 0
    mrr = 0
    
    if pos_rank <= k:
        recall = 1
        mrr = 1 / pos_rank
        ndcg = 1 / np.log2(pos_rank + 1)
        
    return pd.Series({
        'ndcg@5': ndcg,
        'recall@5': recall,
        'mrr@5': mrr,
        'hard_neg_failure_rate': hard_neg_failure_binary, # 0 or 1
        'avg_hard_negs_winning': num_hard_negs_winning,   # 0 to k_hard_negs
        'easy_neg_failure_rate': easy_neg_failure_binary, # 0 or 1
        'avg_easy_negs_winning': num_easy_negs_winning,   # 0 to k_easy_negs
        'positive_rank': pos_rank # absolute rank (1 to total cvs per jds)
    })
