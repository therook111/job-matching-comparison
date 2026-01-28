from src.data_preprocessor import JobsPreprocessor
from src.utils.config_loader import ConfigLoader
import pandas as pd 

config_loader = ConfigLoader()
preprocessor = JobsPreprocessor(config_loader)

if __name__ == '__main__':
    df = pd.read_csv('scripts_output/djinni_jobs_clustered.csv')
    
    df = df[~df['cluster'].isin([4, 5, 13])] # Check cluster_labels.jsonl. Irrelevant clusters.
    
    df = df.groupby('cluster').apply(
        lambda x: x.sample(n=min(len(x), 500), random_state=42)
    ).reset_index(drop=True)
    
    df = preprocessor.process(df)
    
    df.to_csv('scripts_output/djinni_seed_jd.csv', index=False)
