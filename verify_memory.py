import os
import psutil
import time
import threading
from lerobot.datasets.lerobot_dataset import LeRobotDataset

# Global flag to stop monitoring
stop_monitoring = False
peak_memory = 0

def monitor_memory(pid, interval=0.01):
    global peak_memory
    process = psutil.Process(pid)
    while not stop_monitoring:
        mem = process.memory_info().rss / 1024 / 1024
        if mem > peak_memory:
            peak_memory = mem
        time.sleep(interval)

def run_test(desc, func):
    global stop_monitoring, peak_memory
    stop_monitoring = False
    peak_memory = 0
    
    print(f"\n--- {desc} ---")
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_memory, args=(os.getpid(),))
    monitor_thread.start()
    
    start_time = time.time()
    try:
        func()
    finally:
        stop_monitoring = True
        monitor_thread.join()
    
    end_time = time.time()
    print(f"Peak RAM Usage: {peak_memory:.2f} MB")
    print(f"Time taken: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":

    def load_dataset(episodes):
        ds = LeRobotDataset("aractingi/droid_1.0.1", episodes=episodes)
        print(f"Loaded {len(ds)} frames")

    run_test("Loading Subset (episodes=0..499)", lambda: load_dataset(list(range(500))))
    run_test("Loading Subset (episodes=0..4999)", lambda: load_dataset(list(range(5000))))
    run_test("Loading Subset (episodes=0..4999)", lambda: load_dataset(list(range(10000))))