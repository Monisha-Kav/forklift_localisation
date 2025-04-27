import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
import numpy as np

class SensorFusionKalmanNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_kalman_node')

        # State vector: x, y, theta
        self.state = np.array([0.0, 0.0, 0.0])

        # Covariance matrix
        self.P = np.eye(3) * 0.1  # Initial uncertainty

        # Process noise covariance
        self.Q = np.diag([0.01, 0.01, 0.01])

        # Measurement noise covariance
        self.R = np.diag([0.05, 0.05, 0.05])

        self.last_time = self.get_clock().now()

        self.ax = 0.0  # Linear acceleration x
        self.omega = 0.0  # Angular velocity z

        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)

        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(PoseStamped, '/marker_pose', self.marker_callback, 10)

        self.timer = self.create_timer(0.05, self.update_state)  # 20 Hz

    def imu_callback(self, msg):
        self.ax = msg.linear_acceleration.x
        self.omega = msg.angular_velocity.z

    def update_state(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0 or dt > 1:
            return

        # Prediction Step
        x, y, theta = self.state

        # Assume simple motion model: acceleration affects x directly, omega affects theta
        v = self.ax * dt  # Very simple assumption
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = self.omega * dt

        self.state += np.array([dx, dy, dtheta])

        # Jacobian of the motion model (for linearization)
        F = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt],
            [0, 0, 1]
        ])

        # Update covariance
        self.P = F @ self.P @ F.T + self.Q

        self.publish_pose()

    def marker_callback(self, msg):
        measured_x = msg.pose.position.x
        measured_y = msg.pose.position.y

        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        measured_theta = 2 * np.arctan2(qz, qw)

        z = np.array([measured_x, measured_y, measured_theta])

        # Measurement model (direct observation)
        H = np.eye(3)

        # Kalman Gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update step
        innovation = z - H @ self.state
        # Normalize angle to [-pi, pi]
        innovation[2] = np.arctan2(np.sin(innovation[2]), np.cos(innovation[2]))

        self.state = self.state + K @ innovation

        # Update covariance
        I = np.eye(3)
        self.P = (I - K @ H) @ self.P

        self.get_logger().info(f"[CORRECT] Corrected pose: {self.state}")

    def publish_pose(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'

        pose_msg.pose.position.x = float(self.state[0])
        pose_msg.pose.position.y = float(self.state[1])
        pose_msg.pose.position.z = 0.0

        pose_msg.pose.orientation.z = np.sin(self.state[2] / 2)
        pose_msg.pose.orientation.w = np.cos(self.state[2] / 2)

        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionKalmanNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
