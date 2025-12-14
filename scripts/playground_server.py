import cv2
import json
import logging
import numpy as np
import pyrallis
import torch
import torch.nn.functional as F
from dataclasses import dataclass, field
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from typing import Dict, List, Optional, Tuple, Any

from src.world_model.backbone import WorldModelBackbone, WorldModelConfig
from src.rae_dino.rae import RAE
from src.dataset.world_dataset import WorldDatasetConfig, WorldDataset
from src.dataset.lerobot_dataset import LeRobotDatasetConfig, LeRobotDataset
from src.dataset.loader import DataloaderConfig
from src.diffusion.euler_solver import EulerSolver, EulerSolverConfig
from src.diffusion.signal_scheduler import SignalSchedulerConfig
from src.world_model.rollout import rollout_latents
from src.training.utils import set_seed
from src.dataset.common import WorldBatch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PlaygroundConfig:
    checkpoint_path: str = "checkpoints/world_model.pt"
    host: str = "0.0.0.0"
    port: int = 8000
    device: str = "cuda"
    
    dataset: LeRobotDatasetConfig = field(default_factory=LeRobotDatasetConfig)
    context_length: int = 6
    rollout_signal_level: float = 0.9
    sensitivity: float = 1.0
    
    key_mapping: Dict[str, List[float]] = field(default_factory=dict)
    
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    euler_solver: EulerSolverConfig = field(default_factory=EulerSolverConfig)
    signal_scheduler: SignalSchedulerConfig = field(default_factory=SignalSchedulerConfig)
    
    def __post_init__(self):
        if not self.key_mapping:
            self.key_mapping = {
                "w": [0, 1.0], "s": [0, -1.0],
                "a": [1, 1.0], "d": [1, -1.0],
                "q": [2, 1.0], "e": [2, -1.0],
                "arrowup": [3, 1.0], "arrowdown": [3, -1.0],
                "arrowleft": [4, 1.0], "arrowright": [4, -1.0],
                "j": [5, 1.0], "l": [5, -1.0],
                "space": [6, 1.0], "shift": [6, -1.0]
            }

app = FastAPI()
config: PlaygroundConfig = None
model: WorldModelBackbone = None
autoencoder: RAE = None
solver: EulerSolver = None
dataset_backend: Any = None
dataset_stats: Any = None

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>World Model Playground</title>
        <style>
            body { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100vh; margin: 0; background-color: #222; color: white; font-family: sans-serif; }
            #video { width: 640px; height: 640px; border: 2px solid #555; background-color: black; }
            #controls { margin-top: 20px; font-size: 14px; color: #aaa; text-align: center; }
            #status { margin-top: 10px; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>World Model Playground</h1>
        <img id="video" src="" alt="Stream" />
        <div id="status">Connecting...</div>
        <div id="controls">
            <p>Controls: W/S (X), A/D (Y), Q/E (Z) | Arrows (Rot) | J/L (Rot Z) | Space/Shift (Gripper)</p>
            <button onclick="reset()">Reset Episode</button>
        </div>
        <script>
            var ws = new WebSocket("ws://" + window.location.host + "/ws");
            var video = document.getElementById("video");
            var statusDiv = document.getElementById("status");
            
            var activeKeys = new Set();
            
            ws.onopen = function(event) {
                statusDiv.innerText = "Connected";
                startLoop();
            };
            
            ws.onmessage = function(event) {
                var reader = new FileReader();
                reader.onload = function(e) {
                    video.src = e.target.result;
                };
                reader.readAsDataURL(event.data);
            };
            
            ws.onclose = function(event) {
                statusDiv.innerText = "Disconnected";
            };
            
            function startLoop() {
                setInterval(function() {
                    if (ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({
                            "keys": Array.from(activeKeys),
                            "type": "control"
                        }));
                    }
                }, 50);
            }
            
            function reset() {
                if (ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({"type": "reset"}));
                }
            }

            document.addEventListener('keydown', function(event) {
                activeKeys.add(event.key.toLowerCase());
            });

            document.addEventListener('keyup', function(event) {
                activeKeys.delete(event.key.toLowerCase());
            });
        </script>
    </body>
</html>
"""

class DummyLogger:
    def info(self, msg): pass
    def warning(self, msg): pass

def load_model():
    global config, model, autoencoder, solver, dataset_backend, dataset_stats
    
    device = torch.device(config.device)
    
    logger.info("Loading World Model...")
    model = WorldModelBackbone(config.world_model)
    model.to(device)
    
    checkpoint = torch.load(config.checkpoint_path, map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    if "ema_model" in checkpoint:
        state_dict = checkpoint["ema_model"]
        
    new_state_dict = {}
    for k, v in state_dict.items():
        key = k[10:] if k.startswith("_orig_mod.") else k
        new_state_dict[key] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    
    logger.info("Loading Autoencoder...")
    autoencoder = RAE()
    autoencoder.to(device)
    autoencoder.eval()
    
    logger.info("Loading Dataset Metadata...")
    ds = LeRobotDataset(config.dataset, DummyLogger())
    dataset_backend = ds.backend
    dataset_stats = ds.stats
    
    solver = EulerSolver(config.euler_solver, config.signal_scheduler)
    
    logger.info("Ready!")

@app.get("/")
async def get():
    return HTMLResponse(html)

def encode_latents(frames: torch.Tensor, device: torch.device) -> torch.Tensor:
    frames_norm = frames.float() / 255.0
    b, t, c, h, w = frames_norm.shape
    flat = frames_norm.view(b * t, c, h, w)
    with torch.no_grad():
        latents = autoencoder.encode(flat)
    tokens, dim = latents.shape[1], latents.shape[2]
    return latents.view(b, t, tokens, dim)


def decode_and_send_frame(latent: torch.Tensor) -> bytes:
    flat = latent.view(-1, latent.shape[-2], latent.shape[-1])
    with torch.no_grad():
        rec_img = autoencoder.decode(flat)
    img_tensor = rec_img.squeeze(0).cpu().clamp(0, 1)
    img_np = img_tensor.permute(1, 2, 0).numpy()
    img_np = (img_np * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    return buffer.tobytes()


def load_context_frames(start_idx: int, cam_key: str, device: torch.device) -> torch.Tensor:
    context_frames_list = []
    for i in range(config.context_length):
        item = dataset_backend[start_idx + i]
        img = item[cam_key]
        context_frames_list.append(img.unsqueeze(0))
    return torch.cat(context_frames_list, dim=0).unsqueeze(0).to(device)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    device = torch.device(config.device)
    cam_key = list(config.dataset.cameras.keys())[0]
    
    episode_idx = np.random.choice(len(dataset_backend.meta.episodes))
    ep_meta = dataset_backend.meta.episodes[episode_idx]
    start_idx = ep_meta["dataset_from_index"]
    
    context_frames = load_context_frames(start_idx, cam_key, device)
    context_latents = encode_latents(context_frames, device)
    
    rollout_generator = None
    last_clean_frame = None
    
    try:
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message["type"] == "reset":
                episode_idx = np.random.choice(len(dataset_backend.meta.episodes))
                ep_meta = dataset_backend.meta.episodes[episode_idx]
                start_idx = ep_meta["dataset_from_index"]
                context_frames = load_context_frames(start_idx, cam_key, device)
                context_latents = encode_latents(context_frames, device)
                rollout_generator = None
                last_clean_frame = None
                continue
            
            keys = message["keys"]
            action_vec = torch.zeros(1, config.context_length + 1, 7, device=device)
            for key in keys:
                if key in config.key_mapping:
                    dim, val = config.key_mapping[key]
                    action_vec[0, -1, dim] += val * config.sensitivity
            
            if rollout_generator is None:
                rollout_generator = rollout_latents(
                    model,
                    solver,
                    context_latents,
                    config.context_length,
                    future_len=1,
                    rollout_signal_level=config.rollout_signal_level,
                    actions=action_vec,
                )
            else:
                current_latents = torch.cat([context_latents, last_clean_frame], dim=1)
                single_action = action_vec[:, -1:, :]
                rollout_generator = rollout_latents(
                    model,
                    solver,
                    current_latents,
                    config.context_length + 1,
                    future_len=1,
                    rollout_signal_level=config.rollout_signal_level,
                    actions=torch.cat([action_vec[:, :-1, :], single_action], dim=1),
                )
            
            for _, clean_frame, _ in rollout_generator:
                last_clean_frame = clean_frame
                frame_bytes = decode_and_send_frame(clean_frame)
                await websocket.send_bytes(frame_bytes)
                break
                
    except WebSocketDisconnect:
        logger.info("Client disconnected")
        
def main():
    global config
    config = pyrallis.parse(config_class=PlaygroundConfig, config_path="configs/playground.yaml")
    load_model()
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)

if __name__ == "__main__":
    main()
