import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import json
import os
from src.utils.constants import JD_STOP_WORDS
from src.utils.config_loader import ConfigLoader

def main(input_path: str):
    config = ConfigLoader().load_config().get("clustering")
    
    os.makedirs('scripts_output', exist_ok=True)
    
    
    if not os.path.exists(input_path):
        print(f"Error: Could not find input file at {input_path}")
        return
        
    df = pd.read_parquet(input_path)
    
    vectorizer = TfidfVectorizer(
        stop_words=JD_STOP_WORDS, 
        max_features=config["max_features"]
    )
    
    X = vectorizer.fit_transform(df['Long Description'])
    
    kmeans = MiniBatchKMeans(
        n_clusters=config["n_clusters"], 
        random_state=config["random_state"], 
        batch_size=config["batch_size"]
    )
    df['cluster'] = kmeans.fit_predict(X)
    
    order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names_out()
    
    labels_file = 'scripts_output/cluster_labels.jsonl'
    with open(labels_file, 'w', encoding='utf-8') as f:
        for i in range(config["n_clusters"]):
            top_terms = [terms[ind] for ind in order_centroids[i, :6]]
            label = "/".join(top_terms)
            count = int(len(df[df['cluster'] == i]))
            
            print(f"Cluster {i} ({count} JDs): {label}")
            
            line = json.dumps({"id": i, "label": label, "count": count}, ensure_ascii=False)
            f.write(line + '\n')
            
    output_path = 'scripts_output/djinni_jobs_clustered.csv'
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main('thesis resource/JD data.parquet')
