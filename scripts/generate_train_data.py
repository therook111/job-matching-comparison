import pandas as pd
import os
import json
import sys
import asyncio
import concurrent.futures
import random
from tqdm.asyncio import tqdm as async_tqdm # Better async support
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.data_generation.dataset_generator import DatasetGenerator
from src.utils.logger import get_logger
from dotenv import load_dotenv

load_dotenv()
logger = get_logger(__name__)

# CONFIGURATION
MAX_CONCURRENT_REQUESTS = 36
MAX_RETRIES = 3
INITIAL_BACKOFF = 2

async def process_jd(semaphore, executor, generator, jd_text, language, failed_output_path, file_lock):
    """
    Process a single JD with retries and non-blocking architecture.
    """
    async with semaphore:
        loop = asyncio.get_running_loop()
        
        # Retry Loop
        for attempt in range(MAX_RETRIES):
            try:
                await loop.run_in_executor(executor, generator.generate_dataset_entry, jd_text, language)
                return True
                
            except Exception as e:
                if attempt < MAX_RETRIES - 1:
                    sleep_time = INITIAL_BACKOFF * (2 ** attempt)
                    await asyncio.sleep(sleep_time)
                else:
                    error_data = json.dumps({
                        "original_jd": jd_text,
                        "error": str(e)
                    }, ensure_ascii=False) + '\n'
                    
                    # Use a lock to prevent race conditions when writing to file
                    async with file_lock:
                        with open(failed_output_path, 'a', encoding='utf-8') as f:
                            f.write(error_data)
                    return False

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

async def generate_train_data_async():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: return

    input_path = os.path.join("scripts_output", "djinni_train_seed_jd.csv")

    df = pd.read_csv(input_path)
    logger.info("Initializing DatasetGenerator...")
    generator = DatasetGenerator(api_key=api_key)

    # Load already processed JDs to skip
    processed_positive = load_processed_jds(generator.positive_output_path)
    processed_negative = load_processed_jds(generator.negative_output_path)
    processed_jds = processed_positive.union(processed_negative)
    
    logger.info(f"Found {len(processed_jds)} already processed JDs.")

    jd_texts = [
        row["Long Description"] 
        for _, row in df.iterrows() 
        if isinstance(row["Long Description"], str) and row["Long Description"].strip()
    ]
    
    # Filter out already processed JDs
    initial_count = len(jd_texts)
    jd_texts = [text for text in jd_texts if text not in processed_jds]
    logger.info(f"Skipping {initial_count - len(jd_texts)} JDs. Processing {len(jd_texts)} new JDs.")

    failed_output_path = os.path.join("scripts_output", "failed_jds.jsonl")
    if os.path.exists(failed_output_path):
        os.remove(failed_output_path)
    
    # Actual running code
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    file_lock = asyncio.Lock()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS + 5) as executor:
        
        tasks = []
        for jd_text in jd_texts:
            language = "Vietnamese" if random.random() < 0.5 else "English"
            tasks.append(process_jd(semaphore, executor, generator, jd_text, language, failed_output_path, file_lock))
        
        results = await async_tqdm.gather(*tasks, desc="Generating CVs")
        
    success_count = sum(results)
    failure_count = len(results) - success_count
            
    logger.info("Generation complete.")
    logger.info(f"Successfully generated entries: {success_count}")
    logger.info(f"Failed entries: {failure_count}")

if __name__ == "__main__":
    asyncio.run(generate_train_data_async())