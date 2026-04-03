import pandas as pd
from datasets import Dataset

def convert_to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """
    Utility function to convert Confit enriched dataset (.csv) into HF dataset format, to align with
    SentenceTransformers's training API.
    """
    
    grouped_data =[]

    for jd, group in df.groupby("jd"):
        pos_cvs = group[group["match"] == 1]["cv"].tolist()
        neg_cvs = group[group["match"] == 0]["cv"].tolist()
        
        if not pos_cvs:
            continue 
            
        grouped_data.append({
            "anchor": jd,
            "positive": pos_cvs[0], # 1 positive per JD
            "negative": neg_cvs # includes RUM, generative and easy negatives
        })
    
    hf_dataset = Dataset.from_list(grouped_data)
    return hf_dataset

def get_mapping_status(s, seed_jd, seed_map):
    idx = s.find('\n\n## Ideal Candidate Profile\n')

    prefix = str(s)[:idx][100:700]
    suffix = str(s)[:idx][-300:]
    # Get all clusters that match this prefix
    matches = [seed_map[r] for r in seed_jd if prefix in r and suffix in r]
    
    if not matches:
        return "No Match"
    
    # If more than one unique cluster is found, it's ambiguous
    if len(set(matches)) > 1:
        return "Ambiguous"
    
    return matches[0]