# Technical Exercise in Machine Learning and Robotics (v2)

This README is organized into three parts:
1. **Instructions**: how to run the code
2. **Design**: explaning key design decisions
3. **Figures & Media**: screenshots + video demonstrating results from the different steps

## 1. Instructions

### System requirements
- Ubuntu 22.04, ROS2 Humble sourced
- Python 3.10 (ROS2) and Python 3.12 (LeRobot)
- Mujoco Menagerie cloned locally


### 1.1 Build the ROS2 workspace
```bash
mkdir -p ~/ros2_ws/src # standard ROS2 workspace layout

ln -s /path/to/this/repo ~/ros2_ws/src/so101_bridge # symlink repo into src/ folder

cd ~/ros2_ws && colcon build # compiles + installs ROS2 packages

source install/setup.bash # add package to ROS2 runtime env
```

### 1.2 Run the simulator + bridge

```bash
# Terminal 1 — sim + ROS2 bridge 
ros2 run so101_bridge mujoco_bridge --ros-args -p mjcf_path:=$HOME/mujoco_menagerie/robotstudio_so101/scene.xml

# Terminal 2 — demo motion driver
ros2 run so101_bridge trace_letter_i --ros-args \
  -p mjcf_path:=$HOME/mujoco_menagerie/robotstudio_so101/scene.xml \
  -p ee_site_name:=gripperframe
```
Expected result: the arm's end effector traces a capital "I" (top bar, middle stroke, bottom bar) via numerical IK 

### 1.3 Record an episode dataset
```bash
# Terminal 3 — recorder (python 3.10)
ros2 run so101_bridge episode_recorder --ros-args -p mjcf_path:=$HOME/mujoco_menagerie/robotstudio_so101/scene.xml
```
Run alongside Terminals 1–2 **after** the motion driver (Terminal 2) is already running.
Each episode is written to `~/lerobot_recordings/raw/episode_XXX/` (`frames/*.png`, `joint_states.jsonl`, `actions.jsonl`).

### 1.4 Convert to LeRobot dataset format
This step runs in a separate Python 3.12 environment due to LeRobot system requirements

```bash
python3.12 -m venv ~/venvs/lerobot --system-site-packages
source ~/venvs/lerobot/bin/activate
pip install -e ".[dataset]" # from within cloned lerobot repo
python3 convert_to_lerobot.py --repo-id bennytay/so101_wave --output-root ~/lerobot_recordings/dataset
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

The visualizer (`lerobot-dataset-viz`) requires GPU-backed rendering and was run on the host machine (macOS, Apple Silicon) rather than the VM used for Steps 1–3. This is a separate environment from the ROS2/MuJoCo VM setup above.

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

Then copy the dataset from the VM and run the visualizer as described below.


## 2. Key Design Decisions

### Repo / package structure
The repo root doubles as the ROS2 package root (`package.xml`/`setup.py` live at the top level), with the importable Python package one level down in `so101_bridge/`. The ROS2 workspace symlink (`~/ros2_ws/src/so101_bridge`) must point at the **repo root**, not the inner folder, or colcon won't find the package manifest.
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

### Pub/sub topic architecture, not direct calls
The bridge, motion driver, and recorder only ever communicate through `so101/joint_states` / `so101/joint_commands`.

**Reasoning:** this is what makes the recorder a drop-in subscriber rather than something wired into the simulator's internals — any future node (a different controller, a logger, a trained policy) can plug into the same two topics without touching the bridge code. This paid off directly in Step 4, where the trained policy replaces `motion.py` with zero changes to the bridge.

### Task-space IK for scripted motion, not hand-tuned joint angles
The original scripted motion (`wave_motion_node.py`) wrote two hardcoded joint angles directly — pure joint-space scripting, with no IK anywhere in the pipeline (`mujoco_bridge_node.py` applies `so101/joint_commands` straight to `data.ctrl`). That node has since been replaced by `motion.py`, which drives the arm in task space instead: a shape is specified as a handful of (x, y, z) end-effector waypoints, smoothly interpolated with minimum-jerk timing, and converted to joint angles every tick via numerical IK.

**Decision:** implement IK as a small damped-least-squares Jacobian solver (`JacobianIKSolver` in `motion.py`) against a second, kinematics-only MuJoCo model instance loaded from the same MJCF — rather than a scipy-based optimizer or hand-tuned per-waypoint joint angles.

**Reasoning:** `mujoco`/`numpy` are already dependencies, so this adds no new library. Hand-tuning joint angles per waypoint (the original wave-motion approach) doesn't scale to tracing an arbitrary geometric shape and is brittle to re-tune. The IK model instance is intentionally separate from `mujoco_bridge_node.py`'s physics-stepping model — it only ever calls `mj_forward` to compute forward kinematics/Jacobians for a candidate `qpos`, never steps simulated time, so it can't drift from or interfere with the bridge's authoritative simulation.

**Gotcha:** the solver initially called `mujoco.mj_kinematics()` per IK iteration (cheaper than a full `mj_forward`, and sufficient in principle — kinematics alone determines body/site poses and joint anchors/axes). In practice this left the Jacobian exactly zero for every body in the model at the zero pose on a freshly-constructed `MjData`, so the solver could never move away from its `q=0` initial guess. Switching to `mujoco.mj_forward()` fixed it (confirmed with a standalone debug script against the real SO-101 MJCF before rolling the fix into `motion.py`). The extra cost of a full forward pass per iteration is negligible for a 6-DOF arm converging in well under the 50-iteration cap.

**End-effector frame:** the SO-101 MJCF exposes a `gripperframe` site (paired with a `baseframe` site) attached to the `gripper` body — the natural IK target, set via the `ee_site_name` parameter. `motion.py` doesn't hardcode this: it logs every joint/actuator/site/body name at startup and, if `ee_site_name`/`ee_body_name` aren't set, falls back to the last body in the kinematic chain with a warning, so the right frame can be identified for a different MJCF without reading source code.

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

The recording and conversion pipeline (Steps 1.3–1.4 above) completed successfully on the Ubuntu 22.04 ARM64 VM, producing a valid `LeRobotDataset`. The visualization step (`lerobot-dataset-viz`) was run on the host Mac (Apple Silicon, M2) rather than the VM, since it needs GPU-backed rendering that the VM's virtualized graphics stack doesn't support.

Steps taken:
- Copied the converted dataset from the VM to the host: `scp -r bennytay@192.168.64.2:~/lerobot_recordings/dataset ~/dev/lerobot_recordings/dataset_trace_i`
- Cloned `lerobot` fresh on the host and checked out the **same commit** used on the VM during recording/conversion, to avoid any dataset-schema mismatch between versions
- Created a `conda` environment (Python 3.12) and installed `ffmpeg` via `conda install ffmpeg -c conda-forge`, per LeRobot's install docs
- Installed the library with the `dataset_viz` extra (`pip install -e ".[dataset_viz]"`), which bundles `dataset` + `viz` — the correct combination for this CLI
- Ran:
```bash
  lerobot-dataset-viz \
    --repo-id bennytay/so101_trace_i \
    --root ~/dev/lerobot_recordings/dataset_trace_i \
    --episode-index 0 \
    --mode local
```

**Result:** the Rerun viewer launched successfully, rendering the camera feed (`observation.images.cam`), the 6-DOF action trajectory, and joint state — all correctly synced and scrubbable across the episode timeline. See the recording below in [Figures & Media](#3-figures--media).

**Resolved:** initial attempts on the VM hit two platform-specific blockers (software-rasterized rendering under UTM, and CUDA-linked TorchCodec builds with no GPU present) — fixed by running the visualizer natively on the host Mac instead, per the steps above.

**Troubleshooting chain (dataset appeared empty/flat in the viewer):** after the above was working end-to-end, the visualizer still initially showed near-zero, unmoving `observation.state`/`action` plots and a static-looking camera feed. This turned out to be three separate, stacked issues, not one:
1. **Stale converted dataset.** The `dataset_trace_i` being visualized had been built (via `convert_to_lerobot.py`) from a raw recording captured *before* the `motion.py` IK fix above, i.e. while the arm was still stuck at its near-zero home pose. Confirmed directly: every episode's `observation.state` sat at an identical ~0.0002 std, and the parquet file's mtime predated the healthy raw recording's timestamps by ~12 minutes. Fix: re-run `convert_to_lerobot.py` against the current raw recording and re-copy the dataset to the host.
2. **Stale HuggingFace `datasets` cache.** Even after re-converting, the visualizer kept showing the old values — `~/.cache/huggingface/datasets` caches loaded parquet tables and didn't invalidate just because the source file at the same path was overwritten. Fix: `rm -rf ~/.cache/huggingface/datasets/parquet` (plus its `.lock` files) before re-running the visualizer.
3. **Rerun viewer's persisted zoom.** With correct data flowing, the `state` panel's Y-axis was still stuck auto-fit to `observation.state`'s dimension 0 (`shoulder_pan`), which barely rotates for this trace since the "I" is drawn in a single fixed vertical plane — its true range (~0.003 rad) is ~2000x smaller than the other 5 joints (~1.7 to 1.3 rad), so those were plotted far off-screen. Fix: double-click inside the plot panel to reset/autofit the view to all series.

Verified each layer independently along the way (raw JSONL via a debug script, the parquet directly via `pandas`, `meta/stats.json`, and finally the viewer) rather than assuming a fix at one layer meant the whole pipeline was fixed.

### Known limitations
- Capture rate is ~2Hz rather than the intended 10Hz, due to headless rendering overhead.

### Gotchas / lessons learned
- `nano` fails with `Error opening terminal: xterm-ghostty` unless `TERM` is overridden.
- `apt`'s "daemons using outdated libraries" restart prompts are routine during installs — leave checkboxes as-is, select `<Ok>`.
- `pip` inside a `--system-site-packages` venv will report system packages as satisfied even when they're the wrong build for the venv's Python version — use `--ignore-installed` to force a local copy.
- This VM (ARM64, no GPU) repeatedly hits packages that default to CUDA-linked builds (`torch`, `torchcodec`) — always check for a CPU-only install path first for any new ML dependency.
- `episode_recorder` must be started **after** the motion driver is already running — starting it first just records a stationary arm.
- `convert_to_lerobot.py` takes `--repo-id`/`--output-root`/`--raw-dir`/`--force` flags; pass a fresh `--output-root`/`--repo-id` (or `--force`) before converting a new recording rather than overwriting the previous dataset in place.
- A dataset conversion is a snapshot: if a bug upstream (e.g. the IK fix above) is fixed *after* you've already converted, the converted dataset is now stale and needs re-converting — nothing detects or warns about this automatically. What looked like a "near-empty recording" bug when first inspecting `lerobot-dataset-viz` turned out to be exactly this: the dataset predated the fix by ~12 minutes (caught by comparing the parquet file's mtime against the raw recording's own timestamps).
- HuggingFace's `datasets` library caches loaded parquet tables under `~/.cache/huggingface/datasets`, keyed in a way that doesn't reliably invalidate when the source file at the same path is overwritten — after re-converting a dataset in place, clear that cache (`rm -rf ~/.cache/huggingface/datasets/parquet`) before re-running anything that reads it, or you'll keep seeing the old data.
- `episode_recorder_node.py` opens `joint_states.jsonl`/`actions.jsonl` with `'w'` each run (correctly truncating), but previously left old `frames/*.png` files behind from prior runs at the same `episode_XXX` path — now fixed by clearing the `frames/` directory at the start of each episode.

## 3. Figures & Media

**Loading different types of robots into MuJoCo:**

| SO-101 | Franka Panda | Boston Dynamics Spot | Unitree G1 |
|---|---|---|---|
| <img src="images/robot_screenshots/so101.png" width="300"> | <img src="images/robot_screenshots/franka_emika_panda.png" width="300"> | <img src="images/robot_screenshots/boston_dynamics_spot.png" width="300"> | <img src="images/robot_screenshots/unitree_g1.png" width="300"> |

**Making the robot execute a simple motion:**

![SO-101 tracing the letter I](images/robot_gifs/trace_letter_i.gif)

**Obtaining telemetry data:**

![SO-101 telemetry data streaming during the I-trace motion](images/robot_gifs/telemetry_demo.gif)

**Visualizing the recorded dataset (`lerobot-dataset-viz`):**

![Rerun viewer showing the SO-101 camera feed, action trajectory, and joint state for the converted I-trace dataset](images/robot_gifs/lerobot_dataset_viz.gif)

**[INSERT: figure — two-process architecture (Python 3.10 ROS2 side / Python 3.12 LeRobot side)]**

**[INSERT: video — arm driven by the trained ACT policy (Step 4)]**