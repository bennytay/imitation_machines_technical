#!/usr/bin/env python3
"""
Publishes a simple joint trajectory that raises the SO-101 arm and waves.

This is deliberately simple/hardcoded (per the exercise: a quick, working
solution is fine) -- it just ramps one joint up, then oscillates another.

IMPORTANT: the actuator indices used below (1 and 4) are placeholders. Check
the mujoco_bridge_node.py startup log for your model's actual actuator names
and order, then adjust the indices/target angles to match a shoulder-lift
joint and a wrist joint on your specific SO-101 MJCF.

Run:
    ros2 run so101_bridge wave_motion
"""
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class WaveMotion(Node):
    def __init__(self):
        super().__init__('wave_motion')

        self.declare_parameter('num_actuators', 6)
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('shoulder_index', 1)
        self.declare_parameter('wrist_index', 4)
        self.declare_parameter('shoulder_target', -1.0)  # radians
        self.declare_parameter('raise_duration', 3.0)     # seconds
        self.declare_parameter('wave_amplitude', 0.6)     # radians
        self.declare_parameter('wave_freq_hz', 0.8)

        self.n = self.get_parameter('num_actuators').get_parameter_value().integer_value
        rate_hz = self.get_parameter('rate_hz').get_parameter_value().double_value

        self.pub = self.create_publisher(Float64MultiArray, 'so101/joint_commands', 10)
        self.t0 = self.get_clock().now().nanoseconds / 1e9
        self.create_timer(1.0 / rate_hz, self.tick)

        self.get_logger().info(
            f'Publishing wave motion for {self.n} actuators. '
            'Check mujoco_bridge_node log output for actuator names/order '
            'and adjust shoulder_index / wrist_index parameters if needed.'
        )

    def tick(self):
        t = self.get_clock().now().nanoseconds / 1e9 - self.t0

        shoulder_i = self.get_parameter('shoulder_index').get_parameter_value().integer_value
        wrist_i = self.get_parameter('wrist_index').get_parameter_value().integer_value
        shoulder_target = self.get_parameter('shoulder_target').get_parameter_value().double_value
        raise_duration = self.get_parameter('raise_duration').get_parameter_value().double_value
        amp = self.get_parameter('wave_amplitude').get_parameter_value().double_value
        freq = self.get_parameter('wave_freq_hz').get_parameter_value().double_value

        cmd = [0.0] * self.n

        if t < raise_duration:
            cmd[shoulder_i] = shoulder_target * (t / raise_duration)
        else:
            cmd[shoulder_i] = shoulder_target
            wave_t = t - raise_duration
            cmd[wrist_i] = amp * math.sin(2.0 * math.pi * freq * wave_t)

        msg = Float64MultiArray()
        msg.data = cmd
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = WaveMotion()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
