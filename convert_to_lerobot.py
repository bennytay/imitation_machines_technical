#!/usr/bin/env python3
"""
Converts raw episode recordings (produced by episode_recorder_node.py) into
an official LeRobotDataset.

Run this INSIDE the LeRobot venv (Python 3.12), NOT the ROS2/system Python:

    source ~/venvs/lerobot/bin/activate
    pip install pillow  # if not already present in the venv
    python3 convert_to_lerobot.py

NOTE: LeRobot's exact API surface changes between versions. If
LeRobotDataset.create(...) complains about an unexpected/missing keyword
argument, open ~/lerobot/src/lerobot/datasets/lerobot_dataset.py, find the
`create` classmethod, and adjust the call below to match its actual current
signature.
"""
import json
import os
import glob

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

RAW_DIR = os.path.expanduser('~/lerobot_recordings/raw')
REPO_ID = 'bennytay/so101_wave'
OUTPUT_ROOT = os.path.expanduser('~/lerobot_recordings/dataset')
FPS = 2  # actual measured capture rate given software rendering overhead

STATE_DIM = 6
IMG_HEIGHT = 240
IMG_WIDTH = 320


def load_episode(episode_dir):
    frames_dir = os.path.join(episode_dir, 'frames')
    frame_paths = sorted(glob.glob(os.path.join(frames_dir, '*.png')))

    states = []
    with open(os.path.join(episode_dir, 'joint_states.jsonl')) as f:
        for line in f:
            states.append(json.loads(line)['qpos'])

    actions = []
    action_path = os.path.join(episode_dir, 'actions.jsonl')
    if os.path.exists(action_path):
        with open(action_path) as f:
            for line in f:
                actions.append(json.loads(line)['action'])

    n = min(len(frame_paths), len(states), len(actions) if actions else len(states))
    return frame_paths[:n], states[:n], (actions[:n] if actions else states[:n])


def main():
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": None,
        },
        "action": {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": None,
        },
        "observation.images.cam": {
            "dtype": "video",
            "shape": (IMG_HEIGHT, IMG_WIDTH, 3),
            "names": ["height", "width", "channels"],
        },
    }

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        root=OUTPUT_ROOT,
        features=features,
        use_videos=True,
    )

    episode_dirs = sorted(glob.glob(os.path.join(RAW_DIR, 'episode_*')))
    print(f'Found {len(episode_dirs)} raw episodes in {RAW_DIR}')

    for ep_dir in episode_dirs:
        frame_paths, states, actions = load_episode(ep_dir)
        print(f'  {os.path.basename(ep_dir)}: {len(frame_paths)} frames')

        for frame_path, state, action in zip(frame_paths, states, actions):
            img = np.array(Image.open(frame_path))
            dataset.add_frame({
                "observation.state": np.array(state[:STATE_DIM], dtype=np.float32),
                "action": np.array(action[:STATE_DIM], dtype=np.float32),
                "observation.images.cam": img,
                "task": "raise arm and wave",
            })

        dataset.save_episode()

    print(f'Done. Dataset written to {OUTPUT_ROOT}')


if __name__ == '__main__':
    main()
