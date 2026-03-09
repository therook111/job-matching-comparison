from typing import Callable, Dict, List
import numpy as np
import faiss
from src.methods.GNN.schemas import ExtractedEntity
from sklearn.metrics.pairwise import cosine_similarity

EmbedFn = Callable[[List[str]], np.ndarray]


def retrieve_knn(query: str, corpus: faiss.Index, embed_fn: EmbedFn, k: int = 5):
    pass

def calculate_overlap_neighbors_similarity(
    cv_data: List[str], 
    jd_data: List[str],
    corpus: faiss.Index,
    embed_fn: EmbedFn,
    k: int = 10,
    sharpening_factor: float = 4.0,
) -> float:
    
    if not cv_data or not jd_data:
        return 0.0

    cv_knns = {item: set(retrieve_knn(item, corpus, embed_fn, k)) for item in cv_data}
    jd_knns = {item: set(retrieve_knn(item, corpus, embed_fn, k)) for item in jd_data}

    total_overlap_ratio = 0.0
    total_entities = len(cv_data) + len(jd_data)

    for item_cv in cv_data:
        for item_jd in jd_data:
            overlap = cv_knns[item_cv] & jd_knns[item_jd]
            total_overlap_ratio += len(overlap) / k

    return (total_overlap_ratio / total_entities) ** (1 / sharpening_factor)

def calculate_term_similarity(
    cv_data: str,
    jd_data: str,
    embed_fn: EmbedFn,
):
    cv_emb = embed_fn([cv_data])[0]
    jd_emb = embed_fn([jd_data])[0]

    return np.dot(cv_emb, jd_emb) / (
        np.linalg.norm(cv_emb) * np.linalg.norm(jd_emb)
    )

def calculate_l2_dist(
    cv_data: str,
    jd_data: str, 
    ordinal_mapping: Dict[str, int],
    normalize: bool = True
):
    """
    Calculates the L2 distance between two ordinal values.
    
    Args:
        cv_data: The first ordinal value.
        jd_data: The second ordinal value.
        ordinal_mapping: A mapping from ordinal values to their integer representations.
        normalize: Whether to normalize the distance by the maximum possible value.
    
    Output:
        The L2 distance between the two ordinal values
        where 1.0 is the furthest, and 0.0 is identical.

    """
    cv_ordinal = ordinal_mapping.get(cv_data, 0)
    jd_ordinal = ordinal_mapping.get(jd_data, 0)

    diff = abs(cv_ordinal - jd_ordinal)

    if normalize:
        max_possible = max(ordinal_mapping.values())
        return diff / max_possible
    else:
        return diff







            



    