#!/usr/bin/env python3
"""
MuJoCo <-> ROS2 bridge node for the SO-101 arm.

Loads an MJCF model, steps the physics simulation, publishes joint states on
'so101/joint_states', and applies incoming commands from 'so101/joint_commands'
directly as actuator control signals.

Run with the passive viewer (needs the VM's own X11 display, not SSH):

    ros2 run so101_bridge mujoco_bridge --ros-args \
        -p mjcf_path:=/home/bennytay/mujoco_menagerie/robotstudio_so101/scene.xml

Run headless (fine over SSH, no viewer window):

    ros2 run so101_bridge mujoco_bridge --ros-args \
        -p mjcf_path:=/home/bennytay/mujoco_menagerie/robotstudio_so101/scene.xml \
        -p use_viewer:=false

On startup this node logs every joint and actuator name it found in the model --
check that output to know what indices/frames to command in motion.py.
"""
import threading

import numpy as np
import mujoco
import mujoco.viewer

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class MuJoCoBridge(Node):
    def __init__(self):
        super().__init__('mujoco_bridge')

        self.declare_parameter('mjcf_path', '')
        self.declare_parameter('use_viewer', True)
        self.declare_parameter('sim_hz', 200.0)
        self.declare_parameter('publish_hz', 30.0)

        mjcf_path = self.get_parameter('mjcf_path').get_parameter_value().string_value
        if not mjcf_path:
            raise RuntimeError(
                "Set the 'mjcf_path' parameter, e.g. "
                "-p mjcf_path:=/home/bennytay/mujoco_menagerie/robotstudio_so101/scene.xml"
            )

        self.use_viewer = self.get_parameter('use_viewer').get_parameter_value().bool_value
        self.sim_hz = self.get_parameter('sim_hz').get_parameter_value().double_value
        publish_hz = self.get_parameter('publish_hz').get_parameter_value().double_value

        self.get_logger().info(f'Loading MJCF model from {mjcf_path}')
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        # Log discovered joints/actuators so you know what names/indices to command.
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        self.get_logger().info(f'Joints ({self.model.njnt}): {self.joint_names}')
        self.get_logger().info(f'Actuators ({self.model.nu}): {self.actuator_names}')

        # Target ctrl buffer, updated by the command callback, applied every physics step.
        self.ctrl_target = np.zeros(self.model.nu)
        self.lock = threading.Lock()

        # --- ROS pub/sub ---
        self.joint_state_pub = self.create_publisher(JointState, 'so101/joint_states', 10)
        self.create_subscription(
            Float64MultiArray, 'so101/joint_commands', self.command_callback, 10
        )

        # --- Viewer (optional; needs the VM's own display session, not SSH) ---
        self.viewer = None
        if self.use_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        # --- Timers ---
        self.create_timer(1.0 / self.sim_hz, self.physics_step)
        self.create_timer(1.0 / publish_hz, self.publish_joint_state)

    def command_callback(self, msg: Float64MultiArray):
        with self.lock:
            n = min(len(msg.data), self.model.nu)
            self.ctrl_target[:n] = msg.data[:n]

    def physics_step(self):
        with self.lock:
            self.data.ctrl[:] = self.ctrl_target
        mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            if self.viewer.is_running():
                self.viewer.sync()
            else:
                self.get_logger().info('Viewer window closed, shutting down node.')
                rclpy.shutdown()

    def publish_joint_state(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joint_names
        # For a simple arm with no free joints, qpos/qvel line up 1:1 with njnt.
        msg.position = self.data.qpos[: self.model.njnt].tolist()
        msg.velocity = self.data.qvel[: self.model.njnt].tolist()
        self.joint_state_pub.publish(msg)


def main():
    rclpy.init()
    node = MuJoCoBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
