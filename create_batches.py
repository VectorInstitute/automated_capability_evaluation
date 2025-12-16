import csv
import json
import os
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DATASET_DIR = SCRIPT_DIR / "src/task_solve_models/dataset/XFinBench"
SOURCE_CSV = DATASET_DIR / "validation_set.csv"

def create_batches():
    # Read the CSV
    with open(SOURCE_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        all_data = list(reader)

    # Filter/Select ranges
    # Batch 1-5 already exist (0-99). We need 100-199 for Batches 6-10.
    
    batch_size = 20
    start_id = 200
    num_batches = 5
    
    for i in range(num_batches):
        batch_num = 11 + i
        batch_start_idx = start_id + (i * batch_size)
        batch_end_idx = batch_start_idx + batch_size
        
        # Extract rows for this batch
        # Note: 'id' in csv is 'vali_X', we can rely on the order since csv index 0 is vali_0
        # Let's verify via ID matching to be safe
        
        batch_items = []
        for row in all_data:
            row_id_str = row['id'] # e.g. "vali_100"
            try:
                row_idx = int(row_id_str.split('_')[1])
                if batch_start_idx <= row_idx < batch_end_idx:
                    batch_items.append(row)
            except ValueError:
                continue
                
        # Sort by ID just in case
        batch_items.sort(key=lambda x: int(x['id'].split('_')[1]))
        
        if not batch_items:
            print(f"Warning: No items found for batch {batch_num} (IDs {batch_start_idx}-{batch_end_idx})")
            continue

        # Write to JSON
        output_file = DATASET_DIR / f"batch_{batch_num}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(batch_items, f, indent=2)
            
        print(f"Created {output_file.name} with {len(batch_items)} items.")

if __name__ == "__main__":
    create_batches()
