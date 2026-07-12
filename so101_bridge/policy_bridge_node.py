#!/usr/bin/env python3
"""
Relays live observations to a trained ACT policy and republishes its actions
on 'so101/joint_commands' -- the live-inference counterpart to motion.py.

The trained policy only runs in the LeRobot venv (Python 3.12); this node
runs in the ROS2/rclpy environment (Python 3.10), so it can't import or call
the policy directly. Instead it renders the same observation the policy was
trained on (qpos + a camera frame) and sends it over a local TCP socket to
run_policy.py (started separately, in the LeRobot venv), which replies with
the next action. That reply is published to 'so101/joint_commands' exactly
like motion.py does, so mujoco_bridge_node.py needs no changes -- same
pub/sub-only contract as every other node in this package.

Like episode_recorder_node.py, this node owns its own read-only MuJoCo model
purely for rendering: it mirrors the bridge's published qpos via mj_forward
and never steps physics itself.

Run (after run_policy.py is already listening -- see run_policy.py's
docstring for the LeRobot-venv side):
    ros2 run so101_bridge policy_bridge --ros-args \
        -p mjcf_path:=/home/bennytay/mujoco_menagerie/robotstudio_so101/scene.xml
"""
import base64
import json
import socket
import struct
import time

import numpy as np
import mujoco

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


def send_message(sock, obj):
    payload = json.dumps(obj).encode('utf-8')
    sock.sendall(struct.pack('>I', len(payload)) + payload)


def recv_exact(sock, n):
    buf = b''
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_message(sock):
    header = recv_exact(sock, 4)
    if header is None:
        return None
    (length,) = struct.unpack('>I', header)
    payload = recv_exact(sock, length)
    if payload is None:
        return None
    return json.loads(payload.decode('utf-8'))


class PolicyBridge(Node):
    def __init__(self):
        super().__init__('policy_bridge')

        self.declare_parameter('mjcf_path', '')
        self.declare_parameter('num_actuators', 6)
        # Matches the dataset's recording fps (see convert_to_lerobot.py's FPS=2) --
        # the policy was trained on observations spaced this far apart, so querying
        # it faster than this wouldn't reflect the demonstrated timing.
        self.declare_parameter('rate_hz', 2.0)
        self.declare_parameter('camera_name', '')
        self.declare_parameter('image_width', 320)
        self.declare_parameter('image_height', 240)
        self.declare_parameter('policy_host', '127.0.0.1')
        self.declare_parameter('policy_port', 9999)
        self.declare_parameter('connect_timeout_s', 30.0)

        mjcf_path = self.get_parameter('mjcf_path').get_parameter_value().string_value
        if not mjcf_path:
            raise RuntimeError("Set the 'mjcf_path' parameter to your scene.xml")

        self.n = self.get_parameter('num_actuators').get_parameter_value().integer_value
        rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value
        cam_name = self.get_parameter('camera_name').get_parameter_value().string_value
        width = self.get_parameter('image_width').get_parameter_value().integer_value
        height = self.get_parameter('image_height').get_parameter_value().integer_value
        host = self.get_parameter('policy_host').get_parameter_value().string_value
        port = self.get_parameter('policy_port').get_parameter_value().integer_value
        connect_timeout = self.get_parameter('connect_timeout_s').get_parameter_value().double_value

        self.get_logger().info(f'Loading MJCF model from {mjcf_path} (for rendering only)')
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.cam = cam_name if cam_name else -1  # -1 = mujoco's default free camera

        self.latest_qpos = None
        self.create_subscription(JointState, 'so101/joint_states', self.state_cb, 10)
        self.pub = self.create_publisher(Float64MultiArray, 'so101/joint_commands', 10)

        self.get_logger().info(f'Connecting to run_policy.py at {host}:{port} ...')
        self.sock = self._connect_with_retry(host, port, connect_timeout)
        self.get_logger().info('Connected. Starting policy-driven control.')

        self.create_timer(1.0 / rate_hz, self.tick)

    def _connect_with_retry(self, host, port, timeout_s):
        deadline = time.time() + timeout_s
        last_err = None
        while time.time() < deadline:
            try:
                sock = socket.create_connection((host, port), timeout=5.0)
                return sock
            except OSError as e:
                last_err = e
                time.sleep(1.0)
        raise RuntimeError(
            f'Could not connect to run_policy.py at {host}:{port} within {timeout_s}s. '
            f'Make sure run_policy.py is already running. Last error: {last_err}'
        )

    def state_cb(self, msg: JointState):
        self.latest_qpos = np.array(msg.position, dtype=np.float64)

    def tick(self):
        if self.latest_qpos is None:
            return  # nothing published yet, wait

        # Mirror published joint state into our own model, WITHOUT stepping
        # physics -- mj_forward just recomputes derived quantities from qpos,
        # same as episode_recorder_node.py.
        n = min(len(self.latest_qpos), self.model.nq)
        self.data.qpos[:n] = self.latest_qpos[:n]
        mujoco.mj_forward(self.model, self.data)

        self.renderer.update_scene(self.data, camera=self.cam)
        img = self.renderer.render()  # HxWx3 uint8 RGB

        send_message(self.sock, {
            'qpos': self.latest_qpos[: self.n].tolist(),
            'image_b64': base64.b64encode(img.tobytes()).decode('ascii'),
            'image_shape': list(img.shape),
        })

        reply = recv_message(self.sock)
        if reply is None:
            self.get_logger().error('run_policy.py closed the connection. Shutting down.')
            rclpy.shutdown()
            return

        cmd = [0.0] * self.n
        action = reply['action']
        n_copy = min(self.n, len(action))
        cmd[:n_copy] = action[:n_copy]

        msg = Float64MultiArray()
        msg.data = cmd
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = PolicyBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.sock.close()
        except Exception:
            pass
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()


if __name__ == '__main__':
    main()
