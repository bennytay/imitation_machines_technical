# Imitation Machines — Technical Exercise

SO-101 robot arm control and imitation-learning data pipeline: simulator setup, ROS2 integration, LeRobot dataset conversion, and ACT model training.

This README is organized into three parts:
1. **How to Run** — instructions to reproduce the pipeline
2. **Key Design Decisions** — what was decided and why, across the whole project
3. **Figures & Media** — diagrams, images, and video (to be added)

---

## 1. How to Run

### Prerequisites
- Ubuntu 22.04, ROS2 Humble sourced (`source /opt/ros/humble/setup.bash`)
- Python 3.10 (system default, for ROS2) **and** Python 3.12 (for LeRobot) — install 3.12 via the deadsnakes PPA if not already present
- `mujoco_menagerie` cloned locally, with the SO-101 model available at `robotstudio_so101/scene.xml`
- If running headless (no display): `export MUJOCO_GL=osmesa` and `sudo apt install libosmesa6-dev`

### 1.1 Build the ROS2 workspace
```bash
mkdir -p ~/ros2_ws/src
ln -s /path/to/this/repo ~/ros2_ws/src/so101_bridge   # symlink to REPO ROOT, not the inner so101_bridge/ folder
cd ~/ros2_ws
colcon build
source install/setup.bash
```

### 1.2 Run the simulator + bridge
Open three terminals, each with the workspace sourced.

```bash
# Terminal 1 — sim + ROS2 bridge (needs a display, or MUJOCO_GL=osmesa if headless)
ros2 run so101_bridge mujoco_bridge --ros-args -p mjcf_path:=$HOME/mujoco_menagerie/robotstudio_so101/scene.xml

# Terminal 2 — demo motion driver
ros2 run so101_bridge trace_letter_i --ros-args -p mjcf_path:=$HOME/mujoco_menagerie/robotstudio_so101/scene.xml
```
Expected result: the arm's end effector traces a capital "I" (top bar, middle stroke, bottom bar) via numerical IK — driven entirely by the `so101/joint_states` / `so101/joint_commands` topics.

### 1.3 Record an episode dataset
```bash
# Terminal 3 — recorder (same Python 3.10 / ROS2 environment)
ros2 run so101_bridge episode_recorder --ros-args -p output_dir:=$HOME/lerobot_recordings/raw
```
Run alongside Terminals 1–2 while the arm moves. Repeat / restart for multiple episodes. Each episode is written to `~/lerobot_recordings/raw/episode_XXX/` (`frames/*.png`, `joint_states.jsonl`, `actions.jsonl`).

### 1.4 Convert to LeRobot dataset format
This step runs in a **separate Python 3.12 environment** — LeRobot cannot import under ROS2's Python 3.10.

```bash
python3.12 -m venv ~/venvs/lerobot --system-site-packages
source ~/venvs/lerobot/bin/activate
pip install -e ".[dataset]"          # from within your cloned lerobot repo
python convert_to_lerobot.py --raw_dir ~/lerobot_recordings/raw --out_dir ~/lerobot_recordings/dataset
```
Expected result: a validated `LeRobotDataset` at `~/lerobot_recordings/dataset`, with all episodes encoded via AV1/PyAV.

### 1.5 Train and run the ACT policy

Two paths are supported, depending on how much of the pipeline you want to reproduce. **Path B is faster and sufficient to see the trained policy control the arm; Path A additionally verifies the training code itself.**

**Path A — train from scratch**
```bash
# still inside ~/venvs/lerobot (Python 3.12)
python train_act.py --dataset_dir ~/lerobot_recordings/dataset --out_dir ~/lerobot_recordings/checkpoints
```

**Path B — run the provided checkpoint directly (recommended for a quick check)**
```bash
python run_policy.py --checkpoint checkpoints/act_final.ckpt
```
This skips training and loads the checkpoint committed to this repo, then runs inference and publishes actions to `so101/joint_commands` — with Terminal 1 (the bridge) still running, you should see the arm move under the trained policy rather than the scripted `trace_letter_i` driver.

Either path ends the same way: the policy's output actions replace `motion.py` as the thing driving the arm, so Terminal 1 (the bridge) needs to already be running.

### Host Environment Setup (for dataset visualization only)

The visualizer (`lerobot-dataset-viz`) requires GPU-backed rendering and was run on the
host machine (macOS, Apple Silicon) rather than the VM used for Steps 1–3. This is a
separate environment from the ROS2/MuJoCo VM setup above.

```bash
# Clone lerobot on the host, checked out at the same commit used for recording/conversion
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 8a74e0ac6d01706d67fddfed682a09d694d9c8c0

# Create a conda environment (Python 3.12)
conda create -n lerobot-viz python=3.12 -y
conda activate lerobot-viz

# Install ffmpeg via conda (required for this install path, per LeRobot's docs)
conda install ffmpeg -c conda-forge -y

# Install with the dataset_viz extra (dataset loading + rerun-sdk)
pip install -e ".[dataset_viz]"
```

Then copy the dataset from the VM and run the visualizer as described above.


---

## 2. Key Design Decisions

### Repo / package structure
The repo root doubles as the ROS2 package root (`package.xml`/`setup.py` live at the top level), with the importable Python package one level down in `so101_bridge/`. The ROS2 workspace symlink (`~/ros2_ws/src/so101_bridge`) must point at the **repo root**, not the inner folder, or colcon won't find the package manifest.

```
imitation_machines_technical/      # repo root == ROS2 package root
├── package.xml
├── setup.py
├── setup.cfg
├── resource/so101_bridge
├── so101_bridge/
│   ├── mujoco_bridge_node.py      # sim + ROS2 bridge
│   ├── motion.py                  # scripted motion drivers (e.g. trace_letter_i)
│   └── episode_recorder_node.py   # dataset recorder
└── convert_to_lerobot.py          # raw frames -> LeRobotDataset
```

### Pub/sub topic architecture, not direct calls
The bridge, motion driver, and recorder only ever communicate through `so101/joint_states` / `so101/joint_commands`.

**Reasoning:** this is what makes the recorder a drop-in subscriber rather than something wired into the simulator's internals — any future node (a different controller, a logger, a trained policy) can plug into the same two topics without touching the bridge code. This paid off directly in Step 4, where the trained policy replaces `motion.py` with zero changes to the bridge.

### The Python 3.10 / 3.12 split — the central architectural constraint
LeRobot requires **Python 3.12+**. ROS2 Humble's `rclpy` requires **Python 3.10** (the system default). The two cannot share one interpreter.

**Decision:** split the work into two independent OS processes, connected only by plain files on disk — no shared memory, no IPC library. The ROS2/recorder side (3.10) writes raw frames + logs; a separate LeRobot venv (3.12) reads them and builds the dataset, and later, runs training/inference.

**Reasoning:** trying to reconcile the two Python versions in one process would be fragile and hard to debug. Two processes bridged by files means each side uses whatever environment suits it, and the intermediate files are also easier to inspect mid-pipeline than an in-memory queue would be. The same split reappears in Step 4: the trained policy runs in the 3.12 venv, so a thin relay is needed to get its output actions onto `so101/joint_commands`.

**[INSERT: figure of the two-process architecture]**

### Recorder: mirrors state, doesn't re-simulate it
`episode_recorder_node.py` mirrors `qpos` into its own MjModel copy via `mj_forward` — it does **not** step its own physics.

**Reasoning:** the recorder should reflect the authoritative arm state coming from the bridge, not simulate its own copy of physics. Letting it step physics independently risks the recorder's view of the world silently diverging from what the bridge/viewer actually shows — a subtle bug that would only surface later, in the recorded data itself.

### Headless rendering via OSMesa
GLFW (MuJoCo's default renderer) needs a display; the recorder runs over SSH with no `DISPLAY`, so it crashed on startup.

**Decision:** switched to OSMesa software rendering (`MUJOCO_GL=osmesa`, plus `libosmesa6-dev`).

**Trade-off:** software rendering is slow enough per-frame that it directly limited the achievable capture rate — recording ended up at ~2Hz rather than the intended 10Hz (10 episodes, 12–19 frames each, ~120 frames total). This was an overhead problem, not a logic bug.

### Integer FPS for dataset conversion
`convert_to_lerobot.py` sets FPS to an integer (`2`), not the true fractional capture rate.

**Reasoning:** this was a hard library constraint, not a modeling choice — the underlying `av` library's `add_stream` call breaks on a float FPS value.

### Dataset Visualization

The recording and conversion pipeline (Steps 3.1–3.2 above) completed successfully on the
Ubuntu 22.04 ARM64 VM, producing a valid `LeRobotDataset` at `~/lerobot_recordings/dataset`.
The visualization step (`lerobot-dataset-viz`) was run on the host Mac (Apple Silicon, M2)
rather than the VM, since it needs GPU-backed rendering that the VM's virtualized graphics
stack doesn't support.

Steps taken:
- Copied the converted dataset from the VM to the host: `scp -r bennytay@192.168.64.2:~/lerobot_recordings/dataset ~/dev/lerobot_recordings/dataset`
- Cloned `lerobot` fresh on the host and checked out the **same commit** used on the VM during
  recording/conversion, to avoid any dataset-schema mismatch between versions
- Created a `conda` environment (Python 3.12) and installed `ffmpeg` via
  `conda install ffmpeg -c conda-forge`, per LeRobot's install docs
- Installed the library with the `dataset_viz` extra (`pip install -e ".[dataset_viz]"`),
  which bundles `dataset` + `viz` — the correct combination for this CLI
- Ran:
```bash
  lerobot-dataset-viz \
    --repo-id bennytay/so101_wave \
    --root ~/dev/lerobot_recordings/dataset \
    --episode-index 0 \
    --mode local
```

**Result:** the Rerun viewer launched successfully, rendering the camera feed
(`observation.images.cam`), the 6-DOF action trajectory, and joint state — all correctly
synced and scrubbable across the episode timeline.

![Rerun visualizer showing SO-101 camera feed, action trajectory, and joint state](./docs/rerun_viz_episode0.png)

**[INSERT: figure of the visualizer troubleshooting chain]**

**Decision at the time:** abandon the live visualizer rather than keep sinking time into a VM-specific rendering/CUDA issue, and rely on terminal output + direct viewing of raw PNG frames as evidence instead.
**Update:** actively revisiting this — will update this section with the fix (or the final reasoning for staying with the fallback) once resolved.

### Known limitations
- Capture rate is ~2Hz rather than the intended 10Hz, due to headless rendering overhead.
- Dataset visualizer status: see above.

### Gotchas / lessons learned
- `nano` fails with `Error opening terminal: xterm-ghostty` unless `TERM` is overridden.
- `apt`'s "daemons using outdated libraries" restart prompts are routine during installs — leave checkboxes as-is, select `<Ok>`.
- `pip` inside a `--system-site-packages` venv will report system packages as satisfied even when they're the wrong build for the venv's Python version — use `--ignore-installed` to force a local copy.
- This VM (ARM64, no GPU) repeatedly hits packages that default to CUDA-linked builds (`torch`, `torchcodec`) — always check for a CPU-only install path first for any new ML dependency.


## 3. Figures & Media

**Loading different types of robots into MuJoCo:**

| SO-101 | Franka Panda | Boston Dynamics Spot | Unitree G1 |
|---|---|---|---|
| <img src="images/robot_screenshots/so101.png" width="300"> | <img src="images/robot_screenshots/franka_emika_panda.png" width="300"> | <img src="images/robot_screenshots/boston_dynamics_spot.png" width="300"> | <img src="images/robot_screenshots/unitree_g1.png" width="300"> |



**Making the robots execute simple motions:**

**[INSERT: figure — two-process architecture (Python 3.10 ROS2 side / Python 3.12 LeRobot side)]**

**[INSERT: figure — dataset visualizer troubleshooting chain]**

**[INSERT: image(s) of the SO-101 arm in simulation]**

**[INSERT: video — arm driven by scripted "I" trace motion (Steps 1–2)]**

**[INSERT: video — arm driven by the trained ACT policy (Step 4)]**