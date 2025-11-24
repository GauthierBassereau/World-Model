import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# ================= CONFIGURATION =================
# Path to your filtered CSV
CSV_PATH = "/gpfs/helios/home/gauthierbernarda/data/kinetics/annotations/train_no_cut_5s_t30.csv"

# Path to the directory containing the video folders
VIDEO_ROOT = "/gpfs/helios/home/gauthierbernarda/data/kinetics/train"

# SET THIS TO FALSE ONLY WHEN YOU ARE SURE YOU WANT TO DELETE
DRY_RUN = False 
# =================================================

def get_allowed_ids(csv_path):
    """
    Reads the CSV and extracts the unique identifiers (YouTube IDs) 
    that we want to KEEP.
    """
    print(f"Loading whitelist from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Standard Kinetics CSVs usually have a 'youtube_id' column.
    # If your CSV has different headers, adjust 'youtube_id' below.
    if 'youtube_id' not in df.columns:
        raise ValueError(f"Column 'youtube_id' not found in {csv_path}. Available columns: {df.columns}")
    
    # We use a set for O(1) lookup speed
    allowed_ids = set(df['youtube_id'].astype(str))
    print(f"Found {len(allowed_ids)} unique videos to KEEP.")
    return allowed_ids

def cleanup_videos(video_root, allowed_ids):
    deleted_count = 0
    kept_count = 0
    bytes_saved = 0
    
    video_root = Path(video_root)
    
    # Walk through all folders in the train directory
    print(f"Scanning {video_root} for files to remove...")
    
    # We use os.walk to traverse the class folders (e.g., 'playing volleyball')
    for root, dirs, files in os.walk(video_root):
        for file in files:
            file_path = Path(root) / file
            
            # Skip non-video files (optional, adjusts based on your needs)
            if file_path.suffix not in ['.mp4', '.avi', '.mkv']:
                continue

            # LOGIC: Check if the filename contains a whitelisted YouTube ID.
            # Kinetics filenames typically look like: "-0E_XjT-78_000010_000020.mp4"
            # We check if any allowed ID exists within the filename.
            
            should_keep = False
            
            # Optimization: Most filenames START with the youtube_id, 
            # so we can try to split the string to check faster than iterating the whole set.
            # However, strict iteration is safer if formatting varies.
            
            # Heuristic: Extraction logic depends on how torchvision named them.
            # Usually it's {youtube_id}_{start}_{end}.mp4
            # Let's try to extract the ID from the filename string.
            
            # Robust check: is the file part of the allowed set?
            # Since iterating 600k IDs per file is slow, we infer ID from filename.
            # Assuming standard format: ID is the first part before the timestamps.
            
            # If you are unsure of naming convention, use the following verify method:
            # (This assumes the ID is somewhere in the filename)
            file_stem = file_path.stem # filename without extension
            
            # We assume the youtube_id is the first part of the string if split by underscore?
            # Kinetics IDs can contain underscores/dashes.
            # BETTER APPROACH: If you generated train_no_cut.csv, does it have a 'path' column?
            # If not, we have to rely on substring matching.
            
            # Let's assume strict matching isn't possible without the specific 'path' column
            # and use the ID check. To make this fast, we check if the ID 
            # extracted from the file is in our set.
            
            # Adjust this split logic based on your actual filenames!
            # Standard Kinetics: youtube_id_000000_000010.mp4
            # The youtube_id is everything up to the last two underscore sections.
            parts = file_stem.rsplit('_', 2) 
            if len(parts) >= 3:
                potential_id = parts[0]
                if potential_id in allowed_ids:
                    should_keep = True
            else:
                # Fallback for weird names: check if the whole stem is the ID
                if file_stem in allowed_ids:
                    should_keep = True

            if should_keep:
                kept_count += 1
            else:
                if not DRY_RUN:
                    try:
                        file_size = file_path.stat().st_size
                        bytes_saved += file_size
                        os.remove(file_path)
                    except OSError as e:
                        print(f"Error deleting {file_path}: {e}")
                else:
                    # In Dry Run, we just calculate what we WOULD save
                    bytes_saved += file_path.stat().st_size
                
                deleted_count += 1

    print("-" * 30)
    if DRY_RUN:
        print("DRY RUN COMPLETE. NO FILES DELETED.")
    else:
        print("CLEANUP COMPLETE.")
    
    print(f"Videos Kept: {kept_count}")
    print(f"Videos Targeted for Deletion: {deleted_count}")
    print(f"Disk space reclamation: {bytes_saved / (1024**3):.2f} GB")
    print("-" * 30)

if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV file not found at {CSV_PATH}")
    elif not os.path.exists(VIDEO_ROOT):
        print(f"Error: Video directory not found at {VIDEO_ROOT}")
    else:
        ids = get_allowed_ids(CSV_PATH)
        cleanup_videos(VIDEO_ROOT, ids)