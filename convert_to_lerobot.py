#!/usr/bin/env python3
"""
Converts raw episode recordings (produced by episode_recorder_node.py) into
an official LeRobotDataset.

Run this INSIDE the LeRobot venv (Python 3.12), NOT the ROS2/system Python:

    source ~/venvs/lerobot/bin/activate
    pip install pillow  # if not already present in the venv
    python3 convert_to_lerobot.py --repo-id bennytay/so101_wave --output-root ~/lerobot_recordings/dataset

Run with --help for the full list of flags. Defaults match the values used
for the demo recordings, so a bare `python3 convert_to_lerobot.py` still
works for a first run.

NOTE: LeRobot's exact API surface changes between versions. If
LeRobotDataset.create(...) complains about an unexpected/missing keyword
argument, open ~/lerobot/src/lerobot/datasets/lerobot_dataset.py, find the
`create` classmethod, and adjust the call below to match its actual current
signature.
"""
import argparse
import json
import os
import glob

import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

DEFAULT_RAW_DIR = os.path.expanduser('~/lerobot_recordings/raw')
DEFAULT_REPO_ID = 'bennytay/so101_wave'
DEFAULT_OUTPUT_ROOT = os.path.expanduser('~/lerobot_recordings/dataset')
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


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert raw episode recordings into a LeRobotDataset.')
    parser.add_argument('--repo-id', default=DEFAULT_REPO_ID,
                         help=f'Dataset repo_id (default: {DEFAULT_REPO_ID})')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT,
                         help=f'Output dataset root (default: {DEFAULT_OUTPUT_ROOT})')
    parser.add_argument('--raw-dir', default=DEFAULT_RAW_DIR,
                         help=f'Directory containing raw episode_* recordings (default: {DEFAULT_RAW_DIR})')
    parser.add_argument('--force', action='store_true',
                         help='Delete an existing dataset at --output-root before converting, '
                              'instead of failing with FileExistsError.')
    return parser.parse_args()


def main():
    args = parse_args()
    output_root = os.path.expanduser(args.output_root)
    raw_dir = os.path.expanduser(args.raw_dir)

    if args.force and os.path.exists(output_root):
        import shutil
        print(f'--force: removing existing dataset at {output_root}')
        shutil.rmtree(output_root)

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

    try:
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=FPS,
            root=output_root,
            features=features,
            use_videos=True,
        )
    except FileExistsError:
        raise SystemExit(
            f'Dataset already exists at {output_root}. Pass a different '
            f'--output-root/--repo-id for a new dataset, or re-run with --force '
            f'to overwrite it.')

    episode_dirs = sorted(glob.glob(os.path.join(raw_dir, 'episode_*')))
    print(f'Found {len(episode_dirs)} raw episodes in {raw_dir}')

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

    print(f'Done. Dataset written to {output_root}')


if __name__ == '__main__':
    main()
