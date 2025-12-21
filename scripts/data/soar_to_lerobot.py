#!/usr/bin/env python

import argparse
import logging
import shutil
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
import tqdm
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.utils import get_elapsed_time_in_days_hours_minutes_seconds

# --- Configuration ---
FPS = 5
ROBOT_TYPE = "WidowX"

# Define the data schema based on your .npy files
FEATURES = {
    # -- Standard LeRobot Metadata --
    "is_first": {"dtype": "bool", "shape": (1,), "names": None},
    "is_last": {"dtype": "bool", "shape": (1,), "names": None},
    "is_terminal": {"dtype": "bool", "shape": (1,), "names": None},
    "language_instruction": {"dtype": "string", "shape": (1,), "names": None},
    
    # -- Visual Observations --
    # Mapping 'trajectory.mp4' to a main camera view
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },

    # -- State Observations --
    # Mapping 'eef_poses.npy' (Shape: N, 7)
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "axes": ["x", "y", "z", "qx", "qy", "qz", "gripper"],
        },
    },

    # -- Actions --
    # Mapping 'actions.npy' (Shape: N, 7)
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "axes": ["relative_x", "relative_y", "relative_z", "relative_qx", "relative_qy", "relative_qz", "binary_gripper"],
        },
    },



    # -- Episode Metadata --
    # Meta data that are the same for all frames in the episode
    "is_episode_successful": {
        "dtype": "bool",
        "shape": (1,),
        "names": None,
    },
    "robot_id": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
    "date": {
        "dtype": "string",
        "shape": (1,),
        "names": None,
    },
}

def load_video_frames(video_path: Path) -> np.ndarray:
    """Reads all frames from an mp4 file using OpenCV."""
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV) to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return np.array(frames) # Shape: (T, H, W, C)



def generate_soar_frames(traj_path: Path) -> Iterator[dict]:
    # 1. Load Data
    actions_path = traj_path / "actions.npy"
    eef_path = traj_path / "eef_poses.npy"
    video_path = traj_path / "trajectory.mp4"
    lang_path = traj_path / "language_task.txt"
    success_path = traj_path / "success.txt"
    robot_id_path = traj_path / "robot_id.txt"
    date_path = traj_path / "time.txt"

    # Check for required files
    if not actions_path.exists():
        logging.warning(f"Skipping {traj_path}: missing actions.npy")
        return
    if not eef_path.exists():
        logging.warning(f"Skipping {traj_path}: missing eef_poses.npy")
        return
    if not video_path.exists():
        logging.warning(f"Skipping {traj_path}: missing trajectory.mp4")
        return

    try:
        actions = np.load(actions_path).astype(np.float32)
        eef_poses = np.load(eef_path).astype(np.float32)
        frames_rgb = load_video_frames(video_path)
    except Exception as e:
        logging.error(f"Error loading data from {traj_path}: {e}")
        return
    
    # Read language instruction (default to placeholder if missing)
    task_desc = "unknown task"
    if lang_path.exists():
        with open(lang_path,  "r") as f:
            content = f.read().strip()
            if content:  # Only use if non-empty
                task_desc = content
        logging.info(f"Task description for {traj_path.name}: {task_desc}")
    else:
        logging.warning(f"Missing language_task.txt for {traj_path}")

    # Read metadata (same for all frames in the episode)
    is_success = False
    if success_path.exists():
        with open(success_path, "r") as f:
            success_str = f.read().strip().lower()
            is_success = success_str in ["true", "1", "success"]
    
    robot_id = "unknown"
    if robot_id_path.exists():
        with open(robot_id_path, "r") as f:
            content = f.read().strip()
            if content:  # Only use if non-empty
                robot_id = str(content)  # Ensure it's a Python string
    
    date_str = "unknown"
    if date_path.exists():
        with open(date_path, "r") as f:
            content = f.read().strip()
            if content:  # Only use if non-empty
                date_str = str(content)  # Ensure it's a Python string

    # Create metadata dict that will be added to every frame
    frame_meta = {
        "is_episode_successful": np.array([is_success]),
        "robot_id": str(robot_id),  # Ensure Python string
        "date": str(date_str),  # Ensure Python string
    }

    num_frames = len(actions)
    
    # Safety check: ensure video and actions align
    # Note: Sometimes videos have +/- a few frames compared to actions. 
    # We clip to the length of the shortest stream.
    min_len = min(num_frames, len(frames_rgb), len(eef_poses))
    logging.info(f"Processing {traj_path.name}: {min_len} frames")
    


    # 3. Iterate and yield frames
    for i in range(min_len):
        frame = {
            "is_first": np.array([i == 0]),
            "is_last": np.array([i == min_len - 1]),
            "is_terminal": np.array([i == min_len - 1]),
            "language_instruction": task_desc,
            
            "observation.images.image": frames_rgb[i],
            "observation.state": eef_poses[i],
            
            "action": actions[i],

        }
        
        # Add task alias to match LeRobot standard
        frame["task"] = task_desc
        
        # Add episode metadata to every frame (same for all frames in the episode)
        frame.update(frame_meta)
        
        # Debug: Verify task is in frame for first frame
        if i == 0:
            logging.info(f"Frame keys for {traj_path.name}: {list(frame.keys())}")
            if "language_instruction" in frame:
                logging.info(f"Instruction value: '{frame['language_instruction']}'")
            else:
                logging.error(f"INSTRUCTION KEY MISSING in frame for {traj_path}!")
        
        yield frame

def port_soar(raw_dir: Path, repo_id: str, push_to_hub: bool = False):
    # Setup LeRobot Dataset
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features=FEATURES,
    )

    # Find all trajectory folders (recursively find folders named 'traj*')
    # Based on your structure: berkeley_robot_0/.../traj40
    logging.info(f"Scanning {raw_dir} for trajectories...")
    traj_folders = sorted(list(raw_dir.rglob("traj*")))
    
    # Filter to ensure they are directories and look valid (contain actions.npy)
    traj_folders = [p for p in traj_folders if p.is_dir() and (p / "actions.npy").exists()]
    
    num_episodes = len(traj_folders)
    logging.info(f"Found {num_episodes} episodes.")

    start_time = time.time()

    for idx, traj_path in enumerate(tqdm.tqdm(traj_folders)):
        # Optional: Print progress nicely
        if idx % 10 == 0:
            elapsed_time = time.time() - start_time
            d, h, m, s = get_elapsed_time_in_days_hours_minutes_seconds(elapsed_time)
            logging.info(f"Processing {idx}/{num_episodes} | Time: {d}d {h}h {m}m {s:.0f}s")

        try:
            for frame in generate_soar_frames(traj_path):
                lerobot_dataset.add_frame(frame)
            
            lerobot_dataset.save_episode()
        except Exception as e:
            logging.error(f"Error processing {traj_path}: {e}")
            continue

    logging.info("Finalizing dataset...")
    lerobot_dataset.finalize()

    if push_to_hub:
        logging.info("Pushing to Hugging Face Hub...")
        lerobot_dataset.push_to_hub()
    
    logging.info("Done!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Root directory of the downloaded Soar dataset (e.g. ./soar-dataset-local).",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Repository identifier (e.g. 'gauthbern/soar_dataset').",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Upload to hub.",
    )
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    port_soar(**vars(args))

if __name__ == "__main__":
    main()