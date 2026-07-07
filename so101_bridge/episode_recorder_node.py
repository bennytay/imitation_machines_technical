#!/usr/bin/env python3
"""
Records SO-101 telemetry + a rendered camera view into raw per-episode files
on disk. A separate conversion script (run in the LeRobot venv, Python 3.12)
later turns these raw recordings into an official LeRobotDataset.

This node loads its own read-only copy of the MJCF model purely for
rendering -- it does NOT step physics itself. It mirrors whatever joint
positions the bridge node publishes and re-renders a camera view from that
state via mj_forward() (recomputes derived quantities like body poses
without advancing simulated time). This keeps it fully decoupled from
mujoco_bridge_node.py -- no shared Python process needed between the ROS2
side and the LeRobot side.

Usage:
    ros2 run so101_bridge episode_recorder --ros-args \
        -p mjcf_path:=/home/bennytay/mujoco_menagerie/robotstudio_so101/scene.xml \
        -p output_dir:=/home/bennytay/lerobot_recordings/raw \
        -p num_episodes:=10 \
        -p episode_seconds:=8.0

Requires imageio in the SAME Python environment as ROS2/rclpy (system
Python 3.10, NOT the lerobot venv):
    pip3 install imageio
"""
import json
import os
import time

import numpy as np
import mujoco
import imageio.v2 as imageio

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class EpisodeRecorder(Node):
    def __init__(self):
        super().__init__('episode_recorder')

        self.declare_parameter('mjcf_path', '')
        self.declare_parameter('output_dir', os.path.expanduser('~/lerobot_recordings/raw'))
        self.declare_parameter('num_episodes', 10)
        self.declare_parameter('episode_seconds', 8.0)
        self.declare_parameter('camera_name', '')  # empty -> default free camera
        self.declare_parameter('image_width', 320)
        self.declare_parameter('image_height', 240)
        self.declare_parameter('capture_hz', 10.0)

        mjcf_path = self.get_parameter('mjcf_path').get_parameter_value().string_value
        if not mjcf_path:
            raise RuntimeError("Set the 'mjcf_path' parameter to your scene.xml")

        self.output_dir = self.get_parameter('output_dir').get_parameter_value().string_value
        self.num_episodes = self.get_parameter('num_episodes').get_parameter_value().integer_value
        self.episode_seconds = self.get_parameter('episode_seconds').get_parameter_value().double_value
        cam_name = self.get_parameter('camera_name').get_parameter_value().string_value
        width = self.get_parameter('image_width').get_parameter_value().integer_value
        height = self.get_parameter('image_height').get_parameter_value().integer_value
        capture_hz = self.get_parameter('capture_hz').get_parameter_value().double_value

        self.get_logger().info(f'Loading MJCF model from {mjcf_path} (for rendering only)')
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.cam = cam_name if cam_name else -1  # -1 = mujoco's default free camera

        self.latest_qpos = None
        self.latest_action = None

        self.create_subscription(JointState, 'so101/joint_states', self.state_cb, 10)
        self.create_subscription(Float64MultiArray, 'so101/joint_commands', self.action_cb, 10)

        os.makedirs(self.output_dir, exist_ok=True)

        self.episode_idx = 0
        self.frame_idx = 0
        self.episode_start_t = None
        self.episode_dir = None
        self.joint_log = None
        self.action_log = None

        self.start_next_episode()
        self.create_timer(1.0 / capture_hz, self.capture_tick)

    def state_cb(self, msg: JointState):
        self.latest_qpos = np.array(msg.position, dtype=np.float64)

    def action_cb(self, msg: Float64MultiArray):
        self.latest_action = list(msg.data)

    def start_next_episode(self):
        if self.episode_idx >= self.num_episodes:
            self.get_logger().info('All episodes recorded. Shutting down.')
            rclpy.shutdown()
            return

        self.episode_dir = os.path.join(self.output_dir, f'episode_{self.episode_idx:03d}')
        os.makedirs(os.path.join(self.episode_dir, 'frames'), exist_ok=True)
        self.joint_log = open(os.path.join(self.episode_dir, 'joint_states.jsonl'), 'w')
        self.action_log = open(os.path.join(self.episode_dir, 'actions.jsonl'), 'w')
        self.frame_idx = 0
        self.episode_start_t = time.time()
        self.get_logger().info(f'Recording episode {self.episode_idx} -> {self.episode_dir}')

    def capture_tick(self):
        if self.latest_qpos is None:
            return  # nothing published yet, wait

        elapsed = time.time() - self.episode_start_t
        if elapsed >= self.episode_seconds:
            self.joint_log.close()
            self.action_log.close()
            self.episode_idx += 1
            self.start_next_episode()
            return

        # Mirror published joint state into our own model, WITHOUT stepping
        # physics -- mj_forward just recomputes derived quantities from qpos.
        n = min(len(self.latest_qpos), self.model.nq)
        self.data.qpos[:n] = self.latest_qpos[:n]
        mujoco.mj_forward(self.model, self.data)

        self.renderer.update_scene(self.data, camera=self.cam)
        img = self.renderer.render()  # HxWx3 uint8 RGB

        frame_path = os.path.join(self.episode_dir, 'frames', f'{self.frame_idx:06d}.png')
        imageio.imwrite(frame_path, img)

        t = time.time()
        self.joint_log.write(json.dumps({'t': t, 'qpos': self.latest_qpos.tolist()}) + '\n')
        if self.latest_action is not None:
            self.action_log.write(json.dumps({'t': t, 'action': self.latest_action}) + '\n')

        self.frame_idx += 1


def main():
    rclpy.init()
    node = EpisodeRecorder()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()

