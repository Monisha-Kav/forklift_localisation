import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
import numpy as np
import math


class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        self.ax_offset = 0.0  # Will be computed automatically or set manually
        self.bias_buffer = []
        self.bias_samples_needed = 100
        self.bias_computed = False

        self.omega = 0.0

        # State: x, y, theta
        self.state = np.array([0.0, 0.0, 0.0])
        self.vx = 0.0  # Linear velocity along heading

        self.last_time = self.get_clock().now()

        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)
        self.create_subscription(Imu, '/adxl_imu', self.imu_callback, 10)
        self.create_subscription(Imu, '/imu', self.gyro_callback, 10)
        self.create_subscription(PoseStamped, '/marker_pose', self.marker_callback, 10)

        self.timer = self.create_timer(0.05, self.update_state)  # 20 Hz

        self.alpha = 0.8  # Trust in prediction

    def imu_callback(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        self.last_time = current_time

        if dt <= 0 or dt > 1.0:
            return

        raw_ax = msg.linear_acceleration.x

        # --- Auto-calibrate bias at startup ---
        if not self.bias_computed:
            self.bias_buffer.append(raw_ax)
            if len(self.bias_buffer) >= self.bias_samples_needed:
                self.ax_offset = sum(self.bias_buffer) / len(self.bias_buffer)
                self.bias_computed = True
                self.get_logger().info(f"Calibrated ax_offset = {self.ax_offset:.4f}")
            return  # Wait until calibration done before processing further

        # Subtract bias offset
        ax_corrected = raw_ax - self.ax_offset

        # Remove gravity tilt effect (already implemented by you)
        theta = self.state[2]
        ax_world = ax_corrected - 9.8 * math.sin(theta)  # Remove gravity projection

        # Deadband to reject noise
        if abs(ax_world) < 0.2:
            ax_world = 0.0

        # Integrate acceleration to velocity
        self.vx += ax_world * dt

        # Apply friction and clipping to velocity
        friction = 0.05
        self.vx *= (1 - friction * dt)
        self.vx = np.clip(self.vx, -0.5, 0.5)

        # Stop tiny velocity drift
        if ax_world == 0.0 and abs(self.vx) < 0.02:
            self.vx = 0.0

        # Compute new position
        dx = self.vx * math.cos(theta) * dt
        dy = self.vx * math.sin(theta) * dt
        dtheta = self.omega * dt

        self.state += np.array([dx, dy, dtheta])

        self.get_logger().info(
            f"[IMU] raw_ax={raw_ax:.4f} offset={self.ax_offset:.4f} ax_corrected={ax_corrected:.4f} "
            f"ax_world={ax_world:.4f} vx={self.vx:.4f} -> state: {self.state}"
        )

    def gyro_callback(self, msg):
        self.omega = msg.angular_velocity.z

    def update_state(self):
        # This function runs at 20Hz but actual integration mostly in imu_callback
        # You can keep or remove this function if you want only imu_callback updates

        # Optionally you can integrate here, but integration is done in imu_callback already
        pass

    def marker_callback(self, msg):
        measured_x = msg.pose.position.x
        measured_y = msg.pose.position.y

        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        measured_theta = 2.0 * np.arctan2(qz, qw)

        measured_state = np.array([measured_x, measured_y, measured_theta])

        # Correct with weighted average (alpha blending)
        self.state = self.alpha * self.state + (1.0 - self.alpha) * measured_state

        self.get_logger().info(f"[CORRECTION] Corrected pose: {self.state}")

    def publish_pose(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(self.state[0])
        pose_msg.pose.position.y = float(self.state[1])
        pose_msg.pose.position.z = 0.0

        pose_msg.pose.orientation.z = math.sin(self.state[2] / 2.0)
        pose_msg.pose.orientation.w = math.cos(self.state[2] / 2.0)

        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

