import argparse
import logging
import pickle
import zipfile
import numpy as np
from datetime import datetime
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import io

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ROBOT_TYPE = "widowx"
FPS = 5
ACTION_DIM = 7
STATE_DIM = 7

FEATURES = {
    "observation.images.primary": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.wrist": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.side1": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.side2": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (STATE_DIM,),
        "names": {"axes": ["joint_0", "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]},
    },
    "action": {
        "dtype": "float32",
        "shape": (ACTION_DIM,),
        "names": {"axes": ["action_0", "action_1", "action_2", "action_3", "action_4", "action_5", "gripper"]},
    },
    "language_instruction": {
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

def load_pickle_from_zip(zip_ref, path):
    with zip_ref.open(path) as f:
        return pickle.load(f)

def load_image_from_zip(zip_ref, path):
    try:
        with zip_ref.open(path) as f:
            img_bytes = io.BytesIO(f.read())
            im = Image.open(img_bytes)
            return np.asarray(im).astype(np.uint8)
    except Exception as e:
        logging.error(f"Error reading image {path}: {e}")
        return None

def get_sorted_image_list(file_list, traj_path, cam_suffix):
    prefix = f"{traj_path}{cam_suffix}/im_"
    images = [f for f in file_list if f.startswith(prefix) and f.endswith(".jpg")]
    
    try:
        images.sort(key=lambda x: int(x.split("im_")[-1].split(".jpg")[0]))
    except ValueError:
        return []
    return images

def process_trajectory_zip(zip_ref, traj_path, file_list, episode_idx):

    parts = traj_path.strip("/").split("/")
    latency_shift = False
    date_str = "unknown"
    
    for part in parts:
        try:
            date_time = datetime.strptime(part, "%Y-%m-%d_%H-%M-%S")
            date_str = part
            # Bridge data collected before 2021-07-23 has a one-step latency shift, see their github
            if date_time < datetime(2021, 7, 23):
                latency_shift = True
            break
        except ValueError:
            continue

    try:
        obs_data = load_pickle_from_zip(zip_ref, traj_path + "obs_dict.pkl")
        policy_data = load_pickle_from_zip(zip_ref, traj_path + "policy_out.pkl")
    except KeyError:
        logging.warning(f"Missing pickle files in {traj_path}")
        return None

    states = obs_data["full_state"]
    
    # Handle occasional dict vs list format in policy_out
    if len(policy_data) > 0 and isinstance(policy_data[0], dict):
        actions = [x["actions"] for x in policy_data]
    else:
        actions = policy_data

    lang_instruction = ""
    lang_path = traj_path + "lang.txt"
    try:
        with zip_ref.open(lang_path) as f:
            lines = [l.decode("utf-8").strip() for l in f.readlines()]
            lines = [l for l in lines if "confidence" not in l]
            if lines:
                lang_instruction = lines[0]
    except KeyError:
        pass 

    cam_map = {
        "primary": "images0",
        "wrist": "images1",
        "side1": "images2",
        "side2": "images3"
    }
    
    cam_files = {name: get_sorted_image_list(file_list, traj_path, folder) 
                 for name, folder in cam_map.items()}

    if not cam_files["primary"]:
        logging.warning(f"No primary images found in {traj_path}")
        return None

    num_steps = min(len(cam_files["primary"]), len(states), len(actions))
    
    start_idx = 0
    end_idx = num_steps
    
    if latency_shift:
        start_idx = 1
        if num_steps < 2: return None
        
    num_frames = end_idx - start_idx

    black_frame = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(num_frames):
        curr_idx = start_idx + i
        
        # Action index logic
        # If latency shift: we want Action[curr_idx - 1]
        # If no shift: we want Action[curr_idx]
        act_idx = curr_idx - 1 if latency_shift else curr_idx

        # Stop if we run out of actions (due to the shift logic)
        if act_idx >= len(actions):
            break

        frame_data = {
            "observation.state": states[curr_idx].astype(np.float32),
            "action": actions[act_idx].astype(np.float32),
            "language_instruction": lang_instruction,
            "task": lang_instruction,
            "date": date_str,
        }

        for cam_key, file_paths in cam_files.items():
            key_name = f"observation.images.{cam_key}"
            
            if curr_idx < len(file_paths):
                img = load_image_from_zip(zip_ref, file_paths[curr_idx])
                if img is None:
                    frame_data[key_name] = black_frame
                else:
                    frame_data[key_name] = img
            else:
                frame_data[key_name] = black_frame

        yield frame_data

def port_bridge_zip(
    zip_path: Path,
    repo_id: str,
    push_to_hub: bool = False,
):
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type=ROBOT_TYPE,
        fps=FPS,
        features=FEATURES,
    )
    
    logging.info(f"Opening zip file: {zip_path}")
    
    with zipfile.ZipFile(zip_path, 'r') as z:
        all_files = z.namelist()
        
        traj_roots = [f.replace("obs_dict.pkl", "") for f in all_files if f.endswith("obs_dict.pkl")]
        logging.info(f"Found {len(traj_roots)} potential trajectories.")
        
        files_by_traj = {root: [] for root in traj_roots}
        
        logging.info("Indexing files from zip (looking for images0-3)...")
        for f in tqdm(all_files, desc="Indexing"):
            if "/images" in f and "/im_" in f:
                parts = f.split("/images")
                if len(parts) > 1:
                    root_guess = parts[0] + "/"
                    if root_guess in files_by_traj:
                        files_by_traj[root_guess].append(f)

        processed_episodes = 0
        
        for traj_root in tqdm(traj_roots, desc="Processing Trajectories"):
            try:
                traj_files = files_by_traj.get(traj_root, [])
                frames = process_trajectory_zip(z, traj_root, traj_files, processed_episodes)
                
                if frames is None: continue

                frame_count = 0
                for frame in frames:
                    lerobot_dataset.add_frame(frame)
                    frame_count += 1
                
                if frame_count > 0:
                    lerobot_dataset.save_episode()
                    processed_episodes += 1
            
            except Exception as e:
                logging.error(f"Failed to process {traj_root}: {e}")
                continue

    logging.info(f"Successfully processed {processed_episodes} episodes.")
    lerobot_dataset.finalize()

    if push_to_hub:
        lerobot_dataset.push_to_hub(tags=["bridge_data_v2", "widowx", "multi-view"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip-path", type=Path, required=True, help="Path to the downloaded zip file")
    parser.add_argument("--repo-id", type=str, required=True, help="HuggingFace Repo ID")
    parser.add_argument("--push-to-hub", action="store_true")
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    port_bridge_zip(**vars(args))

if __name__ == "__main__":
    main()