import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
import numpy as np
import math


class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # State: [x, y, theta]
        self.state = np.array([0.0, 0.0, 0.0])
        self.vx = 0.0
        self.omega = 0.0
        self.ax_offset = 0.0  # Bias to be auto-calibrated
        self.bias_buffer = []
        self.bias_computed = False
        self.bias_samples_needed = 100

        self.last_time = self.get_clock().now()

        # Publishers and Subscribers
        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)
        self.create_subscription(Imu, '/adxl_imu', self.imu_callback, 10)
        self.create_subscription(Imu, '/imu', self.gyro_callback, 10)
        self.create_subscription(PoseStamped, '/marker_pose', self.marker_callback, 10)

        # Parameters for Bayesian update
        self.alpha = 0.8  # Weight for prediction vs measurement

        # Timer for periodic prediction step
        self.timer = self.create_timer(0.05, self.predict_state)  # 20 Hz

    def imu_callback(self, msg):
        raw_ax = msg.linear_acceleration.x

        # Auto-calibration of bias at startup
        if not self.bias_computed:
            self.bias_buffer.append(raw_ax)
            if len(self.bias_buffer) >= self.bias_samples_needed:
                self.ax_offset = sum(self.bias_buffer) / len(self.bias_buffer)
                self.bias_computed = True
                self.get_logger().info(f"Calibrated ax_offset = {self.ax_offset:.4f}")
            return

        # Subtract offset and apply gravity compensation
        ax_corrected = raw_ax - self.ax_offset
        theta = self.state[2]
        ax_world = ax_corrected - 9.8 * math.sin(theta)

        # Deadband to remove noise
        if abs(ax_world) < 0.2:
            ax_world = 0.0

        self.ax = ax_world  # Save for use in prediction step

    def gyro_callback(self, msg):
        self.omega = msg.angular_velocity.z

    def predict_state(self):
        if not self.bias_computed:
            return  # Wait until bias is computed

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        self.last_time = current_time

        if dt <= 0 or dt > 1.0:
            return

        theta = self.state[2]

        # Integrate acceleration to velocity
        self.vx += self.ax * dt

        # Apply friction
        self.vx *= (1 - 0.05 * dt)
        self.vx = np.clip(self.vx, -0.5, 0.5)

        # Zero out tiny velocities
        if abs(self.vx) < 0.02 and self.ax == 0.0:
            self.vx = 0.0

        # Predict new state using motion model
        dx = self.vx * math.cos(theta) * dt
        dy = self.vx * math.sin(theta) * dt
        dtheta = self.omega * dt

        self.state += np.array([dx, dy, dtheta])
        self.publish_pose()

        self.get_logger().info(f"[PREDICT] ax={self.ax:.3f} vx={self.vx:.3f} -> state: {self.state}")

    def marker_callback(self, msg):
        # Get measurement
        measured_x = msg.pose.position.x
        measured_y = msg.pose.position.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        measured_theta = 2.0 * math.atan2(qz, qw)

        measured_state = np.array([measured_x, measured_y, measured_theta])

        # Bayesian Correction (weighted average)
        self.state = self.alpha * self.state + (1.0 - self.alpha) * measured_state

        self.get_logger().info(f"[CORRECT] marker_pose={measured_state} -> updated_state={self.state}")

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

