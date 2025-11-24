import csv
import os
import sys
import concurrent.futures
from tqdm import tqdm
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import contextlib

# ================= CONFIGURATION =================
INPUT_CSV = '/gpfs/helios/home/gauthierbernarda/data/kinetics/annotations/train.csv'
OUTPUT_CSV = '/gpfs/helios/home/gauthierbernarda/data/kinetics/annotations/train_no_cut_5s_t30.csv'
OUTPUT_CSV = '/gpfs/helios/home/gauthierbernarda/data/kinetics/annotations/train_no_cut_5s_t30.csv'
VIDEO_ROOT = '/gpfs/helios/home/gauthierbernarda/data/kinetics/train' 

CONTENT_THRESHOLD = 30.0
MIN_DURATION_SEC = 5.0
NUM_WORKERS = 16
# =================================================

# --- THE SILENCER ---
@contextlib.contextmanager
def suppress_stderr():
    """
    Redirects stderr to /dev/null at the OS level.
    This captures C-level output, FFmpeg logs, and stubborn Python warnings.
    """
    # Save the original stderr file descriptor
    original_stderr_fd = sys.stderr.fileno()
    
    # Open the null device
    with open(os.devnull, 'w') as devnull:
        # Save the original stderr so we can restore it later
        saved_stderr_fd = os.dup(original_stderr_fd)
        
        try:
            # Redirect stderr to /dev/null
            os.dup2(devnull.fileno(), original_stderr_fd)
            yield
        finally:
            # Restore stderr
            os.dup2(saved_stderr_fd, original_stderr_fd)
            os.close(saved_stderr_fd)

def get_video_path(row):
    label = row['label']
    yt_id = row['youtube_id']
    start = int(row['time_start'])
    end = int(row['time_end'])
    filename = f"{yt_id}_{start:06d}_{end:06d}.mp4"
    return os.path.join(VIDEO_ROOT, label, filename)

def process_video(row):
    """
    Returns: (list_of_new_rows, has_cut_boolean)
    """
    video_path = get_video_path(row)

    if not os.path.exists(video_path):
        return [], False

    try:
        # We wrap the NOISY part in the silencer
        with suppress_stderr():
            video = open_video(video_path)
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=CONTENT_THRESHOLD))
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list(video)

        if not scene_list:
            return [], False

        has_cut = len(scene_list) > 1
        new_rows = []
        original_global_start = int(row['time_start'])

        for i, (start_time, end_time) in enumerate(scene_list):
            duration = end_time.get_seconds() - start_time.get_seconds()
            
            if duration >= MIN_DURATION_SEC:
                offset_start = int(start_time.get_seconds())
                offset_end = int(end_time.get_seconds())

                new_start = original_global_start + offset_start
                new_end = original_global_start + offset_end
                
                clean_row = row.copy()
                clean_row['time_start'] = new_start
                clean_row['time_end'] = new_end
                new_rows.append(clean_row)

        return new_rows, has_cut

    except Exception:
        # If there's a real crash, we simply skip the video
        return [], False

def main():
    print(f"--- Starting Fully Silent Processing ---")
    print(f"Workers: {NUM_WORKERS}")
    
    with open(INPUT_CSV, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames

    with open(OUTPUT_CSV, 'w', newline='') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        total_processed = 0
        total_cuts_found = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
            pbar = tqdm(executor.map(process_video, rows), total=len(rows), unit="vid")

            for result_rows, was_cut in pbar:
                total_processed += 1
                if was_cut:
                    total_cuts_found += 1
                
                if result_rows:
                    writer.writerows(result_rows)
                    f_out.flush() 
                
                if total_processed > 0:
                    cut_percentage = (total_cuts_found / total_processed) * 100
                    pbar.set_postfix({"cut_rate": f"{cut_percentage:.1f}%"})

    print(f"\n--- Done ---")
    if total_processed > 0:
        print(f"Final Cut Rate: {(total_cuts_found/total_processed)*100:.2f}%")

if __name__ == "__main__":
    main()