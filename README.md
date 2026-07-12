# Technical Exercise in Machine Learning and Robotics (v2)

This README is organized into three parts:
1. **Instructions**: how to run the code
2. **Design**: explaning key design decisions
3. **Figures & Media**: screenshots + video demonstrating results from the different steps

<details>
<summary><h2>1. Instructions</h2></summary>

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
python3 convert_to_lerobot.py --repo-id bennytay/so101_trace_i --output-root ~/lerobot_recordings/dataset
```
Expected result: a validated `LeRobotDataset` at `~/lerobot_recordings/dataset`, with all episodes encoded via AV1/PyAV.

### 1.5 Train and run the ACT policy

Two paths are supported, depending on how much of the pipeline you want to reproduce. **Path B is faster and sufficient to see the trained policy control the arm; Path A additionally verifies the training run itself.**

**Path A — train from scratch**

Training needs Python 3.12 + torch, same constraint as dataset conversion/visualization — but unlike those steps, training is CPU-heavy enough that it's worth running on the host Mac (M2, MPS-accelerated) rather than the ARM64 VM, which has no GPU passthrough. This reuses the same pinned `lerobot` clone and the dataset already copied to the host for visualization (see "Host Environment Setup" below — do that first if you haven't) and LeRobot's own training CLI, rather than a hand-rolled training loop:
```bash
# Host Mac, from inside the same lerobot clone used for Host Environment Setup below,
# but a separate Python 3.12 conda env with the training extras (not lerobot-viz)
conda create -n lerobot-train python=3.12 -y
conda activate lerobot-train
cd lerobot && pip install -e ".[training]"   # torch/torchvision pulled in as core deps

lerobot-train \
  --dataset.repo_id=bennytay/so101_trace_i \
  --dataset.root=~/lerobot_recordings/dataset \
  --policy.type=act \
  --policy.device=mps \
  --policy.chunk_size=8 \
  --policy.n_action_steps=8 \
  --output_dir=~/lerobot_recordings/checkpoints/act_trace_i \
  --job_name=act_trace_i \
  --steps=5000 \
  --batch_size=8 \
  --save_freq=1000 \
  --log_freq=50 \
  --wandb.enable=false
```
`--policy.chunk_size`/`--policy.n_action_steps` are overridden down from ACT's default of 100: this dataset's episodes are only 12–15 frames long (10 episodes, 144 frames total, fps=2), so a chunk size of 100 would be almost entirely padding. See "Training the ACT policy" below for why `--steps=5000` and what the loss curve looked like.

This produces `~/lerobot_recordings/checkpoints/act_trace_i/checkpoints/last/pretrained_model/` — a directory (config.json + model.safetensors + the bundled normalization stats), not a single checkpoint file. Copy it into the repo as `checkpoints/act_final/` to use with Path B (the committed one in this repo is tracked via Git LFS, since the weights are ~200MB).

**Path B — run the provided checkpoint directly (recommended for a quick check)**

Running the trained policy live needs a relay across the same Python-version split as everywhere else in this repo (see "Live inference across the Python split" below): `run_policy.py` runs the policy in the LeRobot venv (3.12) and serves it over a local socket; `policy_bridge_node.py` runs as a ROS2 node (3.10) that feeds it observations and republishes its actions to `so101/joint_commands`, replacing `trace_letter_i`/`motion.py` as the thing driving the arm.

```bash
# Terminal 1 (unchanged): the bridge
ros2 run so101_bridge mujoco_bridge --ros-args -p mjcf_path:=$HOME/mujoco_menagerie/robotstudio_so101/scene.xml

# Terminal 2 (LeRobot venv, Python 3.12) -- start this first, it listens for Terminal 3
source ~/venvs/lerobot/bin/activate
python3 run_policy.py --checkpoint checkpoints/act_final

# Terminal 3 (ROS2, Python 3.10) -- start only after Terminal 2 logs "Listening on ..."
ros2 run so101_bridge policy_bridge --ros-args -p mjcf_path:=$HOME/mujoco_menagerie/robotstudio_so101/scene.xml
```
With Terminal 1 still running, you should see the arm move under the trained policy instead of the scripted `trace_letter_i` driver.

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

</details>

<details>
<summary><h2>2. Key Design Decisions</h2></summary>

### Overall Architecture

![split architecture](images/diagrams/code_architecture.png)

**Technical Constraint:** LeRobot requires Python 3.12+ but ROS2 Humble's rclpy requires Python 3.10, hence they can't share the same interpreter.

Therefore, the architecture deliberately splits the work into two independent OS processes, connected only by plain files on disk. The ROS2/recorder side (3.10) writes raw frames + logs; a separate LeRobot venv (3.12) reads them and builds the dataset, and later, runs training/inference.

This is because trying to reconcile the two Python versions in one process would be fragile and hard to debug. Two processes bridged by files means each side uses whatever environment suits it, and the intermediate files are also easier to inspect mid-pipeline than an in-memory queue would be. The same split reappears in Step 4: the trained policy runs in the 3.12 venv, so a thin relay is needed to get its output actions onto `so101/joint_commands`.

### Python 3.10 - ROS2 Nodes:

The pipeline comprises of 4 independent ROS2 nodes. Because none of the nodes call each other directly / know of each other, this offers a host of architectural benefits:

- Each node can be developed/tested/debugged in isolation
- Swapping or extending one aspect doesn't require editing the code of the others
- Multiple subscribers can observe the same data without conflict 
- Each node can fail without taking the others down


1: `mujoco_bridge_node.py` (the bridge)

This loads SO101's MJCF model into MuJoCo and steps the physics simultation. It is responsible for simulating the arm, and rendering the viewer window. It exposes the arm's current joint position to `so101/joint_states` and applies what it receives thru `so101/joint_commands` directly to the sim actuators. 

2: `motion.py` (motion driver)

This generates a scripted trajectory (i.e. a capital I shape) as a sequence of end-effector waypoints; solves joint angles needed to reach the waypoint thru its own IK solver; publishes the resulting joint angles to `so101/joint_commands`.

3: `episode_recorder_node.py` (recorder)

This subscribes to both `so101/joint_states` and `so101/joint_commands` and logs what it passively observes to the disk for a fixed duration per episode. 

4: `policy_bridge_node.py` (live policy relay)

This is `motion.py`'s Step-4 counterpart: same publish contract (`Float64MultiArray` on `so101/joint_commands`), but the "motion" comes from a trained ACT policy instead of a scripted IK trajectory. It mirrors the recorder's rendering setup (its own read-only MuJoCo model, subscribed to `so101/joint_states`, mirrored via `mj_forward`) to reconstruct the same observation the policy was trained on, relays it to `run_policy.py` over a local socket, and publishes back whatever action comes back.

> NOTE: The bridge, motion driver, recorder, and policy relay only ever communicate through `so101/joint_states` / `so101/joint_commands`.
>
> This is what makes the recorder (and later the policy relay) drop-in subscribers rather than something wired into the simulator's internals — any future node (a different controller, a logger, a trained policy) can plug into the same two topics without touching the bridge code. This paid off directly in Step 4, where the trained policy replaces `motion.py` with zero changes to `mujoco_bridge_node.py`.

### Implementing Inverse Kinematics

![inverse kins flowchart](images/diagrams/inverse_kins_diagram.png)

`motion.py` drives the arm through task-space control, with the desired motion described as a handful of 3D waypoints, with inverse kinematics converting each target position into the joint angles needed to get there.This process is recalculated every tick. 

This was implemented through the construction of a small Jacobian-based solver `JacobianIKSolver` in `motion.py` which is built upon the `mujoco`/`numpy` tools, and runs against a second physics-free copy of the robot model such that the trial and error calculation space doesn't interfere with the actual rendered simultation. Instead of hardcoding which specific point on the robot the inverse kinematics system aims for, the code auto discoveres and logs all robot's joint names at startup so right target can be identified for different robot model without reading through source code. 

### Headless rendering via OSMesa

Note that the recorder needs its own render pass because the LeRobot dataset requires saved camera image files per timestep compared to the live on-screen display from the MuJoCo bridge's viewer window. Thus, the recorder holds a separate MuJoCo instnace (mirroring the bridge's state) to render + write frames to disk.

The recorder uses  OSMesa software rendering (`MUJOCO_GL=osmesa`, plus `libosmesa6-dev`). This enables the recorder to run headless over SSH without crashing, because OSMesa only requires CPU cycles. In contrast, MuJoCo's default renderer GLFW needs a real display, which SSH sessions by default don't have. 

Choosing OSMesa was most practical for my hardware setup, as the VM used on my M2 Macbook Air (UTM) has no GPU passthrough, meaning I couldn't use EGL, for instance, which needs an actual GPU to be present.



### Integer FPS for dataset conversion

In real world timing, the true capture rate is fractional (e.g. 2.35Hz, 1.87Hz). But `convert_to_lerobot.py` needs to set a clear frame rate for the video when it builds the dataset.

The `av` library used for this has a function called `add_stream`. However this requires FPS be an integer, so the choice was made to set it to `2` rather than the true fractional rate.

### Lerobot Dataset: Visualization

![lerobot dataset visualisation](images/diagrams/lerobot_viz_ss.png)

Initially, I tried to run the LeRobot datset visualisation tool on the VM, however it failed due to CUDA which doesn't exist on non-GPU. The solution was to instead copy the dataset from the VM to the host laptop (my m2 macbook) and run the visualisation tool through there.

Summary of process: The recording and conversion pipeline (Steps 1.3–1.4 above) completed successfully on the Ubuntu 22.04 ARM64 VM, producing a valid `LeRobotDataset`. The visualization step (`lerobot-dataset-viz`) was run on the host Mac (Apple Silicon, M2) rather than the VM, since it needs GPU-backed rendering that the VM's virtualized graphics stack doesn't support.

### Training the ACT policy

**Decision:** train via LeRobot's own `lerobot-train` CLI (`--policy.type=act`) rather than a hand-rolled training loop, on the host Mac (M2, MPS) rather than the ARM64 VM used for Steps 1–3.

**Reasoning:** the exercise brief itself points at "existing LeRobot scripts," and `lerobot-train` already gets normalization, checkpointing, and dataloading right — a custom loop would just be re-implementing that, with more room for subtle bugs (e.g. training-time vs. inference-time normalization drifting apart). Training is far more CPU-intensive than dataset conversion/visualization, and the VM has no GPU passthrough, so it runs on the host instead, reusing the dataset already copied there for visualization (see "Host Environment Setup" above) and the same pinned `lerobot` commit.

`--policy.chunk_size`/`--policy.n_action_steps` were dropped from ACT's default of 100 to 8: this dataset's 10 episodes are only 12–15 frames long (144 frames total at fps=2), so a chunk size of 100 would be predicting almost nothing but padding repeats of the last action. `--steps=5000` was picked after a 10-step smoke test showed ~0.2s/step steady-state throughput on MPS (~5 steps/sec) — at that rate 5000 steps is ~30 minutes, comfortably inside the exercise's "train for about an hour" framing while giving a much more convincing curve than the first back-of-envelope guess of 2000 steps.

**Metrics tracked, and why:**
- **`l1_loss`** — L1 distance between the policy's predicted action chunk and the ground-truth demonstrated actions. This is the core imitation signal: a downward trend is direct evidence the policy is fitting the demonstrated trajectory, independent of the VAE machinery below.
- **`kld_loss`** — KL divergence between the CVAE encoder's latent posterior and its standard-normal prior (ACT trains as a conditional VAE). A high value means the encoder is leaning heavily on extra information smuggled through the latent rather than the observation; watching it fall tells you the latent is settling down rather than being (ab)used as a shortcut.
- **`loss`** — the combined objective actually being optimized (`l1_loss + kl_weight * kld_loss`, `kl_weight=10.0`).
- **`grad_norm`** (`grdn` in the logs) — a stability sanity check; a spike or NaN here would flag a bad learning rate or a data issue before wasting the rest of the training budget on a broken run.

`lerobot-train` only logs the combined `loss`/`grad_norm`/`lr` to stdout by default — the `l1_loss`/`kld_loss` split is only surfaced through Weights & Biases, which this run intentionally skipped (`--wandb.enable=false`, to avoid requiring an account). To still get the breakdown, each saved checkpoint (every 1000 steps) was reloaded afterward and run through one forward pass over the full dataset to recompute `l1_loss`/`kld_loss` directly — a legitimate reconstruction since it's the exact same `ACTPolicy.forward()` call `lerobot-train` itself uses internally.

**What was observed:**

| step | total loss | l1_loss | kld_loss |
|---|---|---|---|
| 1000 | 1.751 | 0.185 | 0.157 |
| 2000 | 1.049 | 0.151 | 0.090 |
| 3000 | 0.598 | 0.129 | 0.047 |
| 4000 | 0.375 | 0.120 | 0.025 |
| 5000 | 0.264 | 0.117 | 0.015 |

(stdout's own running `loss` tracked this closely throughout, e.g. `loss:0.270` at step 5000 vs. the recomputed 0.264 — the small gap is just dropout being active for the stdout figure's live batch vs. this table's full-dataset re-evaluation.)

`l1_loss` fell steadily and kept improving through the full run (0.185 → 0.117) — direct evidence the policy is learning to reproduce the demonstrated trajectory, exactly the "sufficient that the model starts to learn" bar the exercise sets, without claiming full convergence on a 144-frame dataset. `kld_loss` collapsed an order of magnitude (0.157 → 0.015), meaning the latent posterior converged toward the prior rather than continuing to encode extra shortcut information — a well-behaved CVAE, not a sign of trouble (`kld_loss` decaying while `l1_loss` also keeps improving is the healthy pattern; `kld_loss` bottoming out early while `l1_loss` stalls would instead suggest posterior collapse).

The trained checkpoint is committed at `checkpoints/act_final/` via Git LFS (`model.safetensors` alone is ~200MB, over GitHub's 100MB push limit for a plain commit).

### Live inference across the Python split

**Technical Constraint:** same 3.10/3.12 split as everywhere else in this repo, but this time it has to work in both directions, live: the trained policy only runs in the LeRobot venv (3.12), while the arm is only reachable via `rclpy` (3.10-only), and unlike Steps 1–4's file-based handoff, there's no "convert once, read later" — a running policy needs an observation and has to return an action on every tick.

**Decision:** a small stdlib-only loopback TCP socket between two new pieces — `policy_bridge_node.py` (ROS2 node, 3.10) and `run_policy.py` (3.12) — using a length-prefixed JSON protocol, rather than a message broker like ZeroMQ or ROS2's own multi-language bridging.

**Reasoning:** `policy_bridge_node.py` renders the same observation shape the policy was trained on (mirrors `episode_recorder_node.py`'s read-only-model rendering setup) and publishes whatever action comes back exactly like `motion.py` does — so `mujoco_bridge_node.py` needs zero changes, preserving the topic-only decoupling described above. A blocking request/response loop is sufficient because the system already runs at only a few Hz due to OSMesa's software rendering overhead (see below), so there's no latency budget a synchronous socket call would violate. ZeroMQ was considered and rejected: it would mean installing `pyzmq` in both venvs for a plain 1:1 request/response pattern that a ~40-line stdlib socket wrapper already handles, which doesn't match this repo's existing bias toward not adding a dependency when the standard library covers it (same reasoning as the `JacobianIKSolver` decision above).

**Gotcha:** loading a checkpoint back with `ACTPolicy.from_pretrained()` surfaced two non-obvious issues, both in `run_policy.py`:
1. LeRobot's own `pretrained.py` does `import packaging` and then reads `packaging.version.parse(...)` — that only works if something else already imported the `packaging.version` submodule first. `run_policy.py` adds an explicit `import packaging.version` up top to avoid an `AttributeError` at load time.
2. `policy.to(device)` only moves the model's existing parameters. `policy.config.device` (saved as `"mps"`, since training ran on the host Mac) is read directly by parts of the model that create fresh tensors on the fly, and separately by the saved preprocessor pipeline's own device step — both have to be overridden explicitly (`policy.config.device = args.device` and `make_pre_post_processors(..., preprocessor_overrides={'device_processor': {'device': args.device}})`) or inference silently tries to mix `mps` and `cpu` tensors and crashes.

</details>

<details>
<summary><h2>3. Figures & Media</h2></summary>

**Loading different types of robots into MuJoCo:**

| SO-101 | Franka Panda |
|---|---|
| <img src="images/robot_screenshots/so101.png" width="400"> | <img src="images/robot_screenshots/franka_emika_panda.png" width="400"> |
| **Boston Dynamics Spot** | **Unitree G1** |
| <img src="images/robot_screenshots/boston_dynamics_spot.png" width="400"> | <img src="images/robot_screenshots/unitree_g1.png" width="400"> |

**Making the robot execute a simple motion:**

![SO-101 tracing the letter I](images/robot_gifs/trace_letter_i.gif)

**Obtaining telemetry data:**

![SO-101 telemetry data streaming during the I-trace motion](images/robot_gifs/telemetry_demo.gif)

**Visualizing the recorded dataset (`lerobot-dataset-viz`):**

![Rerun viewer showing the SO-101 camera feed, action trajectory, and joint state for the converted I-trace dataset](images/robot_gifs/lerobot_dataset_viz.gif)

**[INSERT: figure — two-process architecture (Python 3.10 ROS2 side / Python 3.12 LeRobot side)]**

**[INSERT: video — arm driven by the trained ACT policy (Step 4)]**

</details>