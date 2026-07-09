#!/usr/bin/env python3
"""
Scripted end-effector trajectories for the SO-101 arm.

This module loads its own lightweight MuJoCo model instance purely for
kinematics (forward kinematics + Jacobians) so it can drive the arm in
task space (x, y, z waypoints) instead of hand-tuned joint angles. It does
NOT touch the physics-stepping model owned by mujoco_bridge_node.py -- it
only reads the same MJCF file to compute joint angles, then publishes those
angles as actuator targets on 'so101/joint_commands', same publishing
pattern as the original scripted motion driver.

Currently implements one trajectory: tracing a capital letter "I" (with
top/bottom serif bars) in a vertical plane in front of the robot. Future
scripted trajectories should live in this same file as additional
Node subclasses / waypoint functions, reusing JacobianIKSolver.

Run:
    ros2 run so101_bridge trace_letter_i --ros-args \
        -p mjcf_path:=/home/bennytay/mujoco_menagerie/robotstudio_so101/scene.xml

On startup this node logs every joint/actuator/site/body name it found in
the model -- check that output if the 'ee_body_name' / 'ee_site_name'
parameters need adjusting, same as you'd check mujoco_bridge_node's log for
actuator names/order.
"""
import numpy as np

import mujoco

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


def minimum_jerk_scale(s: float) -> float:
    """Quintic time-scaling s(t) in [0, 1] -> [0, 1] with zero vel/accel at the ends."""
    s = max(0.0, min(1.0, s))
    return 6.0 * s**5 - 15.0 * s**4 + 10.0 * s**3


def letter_i_waypoints(center, width, height):
    """
    Six (x, y, z) waypoints tracing a capital "I" with top/bottom serif bars,
    in the order: top-left -> top-right -> top-middle -> bottom-middle ->
    bottom-left -> bottom-right (then loops back to top-left).

    `center` is (x, y, z) in the model's base frame; the letter is drawn in
    whichever two axes vary (see TraceLetterI.plane), the third held fixed.
    """
    cx, cy, cz = center
    half_w = width / 2.0
    half_h = height / 2.0
    return [
        (cx - half_w, cy, cz + half_h),  # top-left
        (cx + half_w, cy, cz + half_h),  # top-right
        (cx, cy, cz + half_h),           # top-middle
        (cx, cy, cz - half_h),           # bottom-middle
        (cx - half_w, cy, cz - half_h),  # bottom-left
        (cx + half_w, cy, cz - half_h),  # bottom-right
    ]


class JacobianIKSolver:
    """
    Small damped-least-squares numerical IK solver against a MuJoCo model,
    tracking a target 3D position for a chosen body or site. Reusable across
    any future task-space scripted trajectory in this file.
    """

    def __init__(self, model, data, ee_body_id=None, ee_site_id=None,
                 damping=0.05, max_iters=50, tol=1e-4):
        if (ee_body_id is None) == (ee_site_id is None):
            raise ValueError('Provide exactly one of ee_body_id or ee_site_id')
        self.model = model
        self.data = data
        self.ee_body_id = ee_body_id
        self.ee_site_id = ee_site_id
        self.damping = damping
        self.max_iters = max_iters
        self.tol = tol
        self._jacp = np.zeros((3, model.nv))

    def _forward(self, q):
        self.data.qpos[: self.model.nq] = q
        mujoco.mj_kinematics(self.model, self.data)

    def _ee_pos(self):
        if self.ee_site_id is not None:
            return self.data.site_xpos[self.ee_site_id].copy()
        return self.data.xpos[self.ee_body_id].copy()

    def _jacobian(self):
        if self.ee_site_id is not None:
            mujoco.mj_jacSite(self.model, self.data, self._jacp, None, self.ee_site_id)
        else:
            mujoco.mj_jacBody(self.model, self.data, self._jacp, None, self.ee_body_id)
        return self._jacp

    def solve(self, target_pos, q_init):
        """Returns joint angles (radians) that place the EE at target_pos, warm-started from q_init."""
        q = np.array(q_init, dtype=float).copy()
        target_pos = np.asarray(target_pos, dtype=float)

        for _ in range(self.max_iters):
            self._forward(q)
            err = target_pos - self._ee_pos()
            if np.linalg.norm(err) < self.tol:
                break
            jacp = self._jacobian()
            jjt = jacp @ jacp.T + (self.damping ** 2) * np.eye(3)
            dq = jacp.T @ np.linalg.solve(jjt, err)
            q += dq
            self._clip_to_joint_limits(q)

        return q

    def _clip_to_joint_limits(self, q):
        for j in range(self.model.njnt):
            if self.model.jnt_limited[j]:
                lo, hi = self.model.jnt_range[j]
                q[j] = min(max(q[j], lo), hi)


class TraceLetterI(Node):
    """Drives the SO-101 end effector through a capital-'I' trace via numerical IK."""

    def __init__(self):
        super().__init__('trace_letter_i')

        self.declare_parameter('mjcf_path', '')
        self.declare_parameter('num_actuators', 6)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('ee_site_name', '')
        self.declare_parameter('ee_body_name', '')
        self.declare_parameter('center', [0.25, 0.0, 0.2])  # (x, y, z), base frame, meters
        self.declare_parameter('plane', 'xz')  # which two axes the letter is drawn in
        self.declare_parameter('width', 0.1)
        self.declare_parameter('height', 0.15)
        self.declare_parameter('segment_duration', 1.5)  # seconds per waypoint-to-waypoint leg

        mjcf_path = self.get_parameter('mjcf_path').get_parameter_value().string_value
        if not mjcf_path:
            raise RuntimeError(
                "Set the 'mjcf_path' parameter, e.g. "
                "-p mjcf_path:=/home/bennytay/mujoco_menagerie/robotstudio_so101/scene.xml"
            )

        self.n = self.get_parameter('num_actuators').get_parameter_value().integer_value
        rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value
        self.plane = self.get_parameter('plane').get_parameter_value().string_value
        self.segment_duration = self.get_parameter('segment_duration').get_parameter_value().double_value

        self.get_logger().info(f'Loading MJCF model from {mjcf_path} for kinematics')
        self.model = mujoco.MjModel.from_xml_path(mjcf_path)
        self.data = mujoco.MjData(self.model)

        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        site_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SITE, i)
            for i in range(self.model.nsite)
        ]
        body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            for i in range(self.model.nbody)
        ]
        self.get_logger().info(f'Joints ({self.model.njnt}): {joint_names}')
        self.get_logger().info(f'Actuators ({self.model.nu}): {actuator_names}')
        self.get_logger().info(f'Sites ({self.model.nsite}): {site_names}')
        self.get_logger().info(f'Bodies ({self.model.nbody}): {body_names}')

        self.ee_body_id, self.ee_site_id = self._resolve_ee_frame(site_names, body_names)

        self.ik = JacobianIKSolver(
            self.model, self.data, ee_body_id=self.ee_body_id, ee_site_id=self.ee_site_id
        )

        center = tuple(self.get_parameter('center').get_parameter_value().double_array_value)
        width = self.get_parameter('width').get_parameter_value().double_value
        height = self.get_parameter('height').get_parameter_value().double_value
        self.waypoints = letter_i_waypoints(center, width, height)
        self.num_waypoints = len(self.waypoints)

        self.q_current = np.zeros(self.model.nq)

        self.pub = self.create_publisher(Float64MultiArray, 'so101/joint_commands', 10)
        self.t0 = self.get_clock().now().nanoseconds / 1e9
        self.create_timer(1.0 / rate_hz, self.tick)

        self.get_logger().info(
            f'Tracing letter "I" (width={width}, height={height}, center={center}, '
            f'plane={self.plane}) for {self.n} actuators.'
        )

    def _resolve_ee_frame(self, site_names, body_names):
        """Pick the end-effector body or site to drive with IK, per the ee_*_name params."""
        site_name = self.get_parameter('ee_site_name').get_parameter_value().string_value
        body_name = self.get_parameter('ee_body_name').get_parameter_value().string_value

        if site_name:
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                raise RuntimeError(f"ee_site_name '{site_name}' not found. Sites: {site_names}")
            self.get_logger().info(f"Using site '{site_name}' as end effector.")
            return None, site_id

        if body_name:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                raise RuntimeError(f"ee_body_name '{body_name}' not found. Bodies: {body_names}")
            self.get_logger().info(f"Using body '{body_name}' as end effector.")
            return body_id, None

        # No override given: default to the last body in the kinematic chain and
        # log it clearly, since this is a guess -- set ee_body_name/ee_site_name
        # above if this isn't the right link.
        body_id = self.model.nbody - 1
        self.get_logger().warn(
            f"No ee_site_name/ee_body_name set. Defaulting to last body "
            f"'{body_names[body_id]}' (id {body_id}) as the end effector -- "
            f"set ee_site_name or ee_body_name if this is wrong."
        )
        return body_id, None

    def _target_position(self, t):
        cycle_t = t % (self.segment_duration * self.num_waypoints)
        seg_idx = int(cycle_t // self.segment_duration)
        seg_t = (cycle_t - seg_idx * self.segment_duration) / self.segment_duration
        s = minimum_jerk_scale(seg_t)

        p0 = np.array(self.waypoints[seg_idx])
        p1 = np.array(self.waypoints[(seg_idx + 1) % self.num_waypoints])
        return p0 + s * (p1 - p0)

    def tick(self):
        t = self.get_clock().now().nanoseconds / 1e9 - self.t0
        target = self._target_position(t)

        self.q_current = self.ik.solve(target, self.q_current)

        cmd = [0.0] * self.n
        n_copy = min(self.n, len(self.q_current))
        cmd[:n_copy] = self.q_current[:n_copy].tolist()

        msg = Float64MultiArray()
        msg.data = cmd
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = TraceLetterI()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
