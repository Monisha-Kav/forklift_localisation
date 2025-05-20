import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
import math

class KalmanFilterNode(Node):
    def __init__(self):
        super().__init__('kalman_sensor_fusion_node')

        # [x, y, theta, v, omega]
        self.state = np.zeros((5, 1))
        self.P = np.eye(5) * 0.1

        self.last_time = self.get_clock().now()

        # Process noise
        self.Q = np.diag([0.01, 0.01, 0.01, 0.05, 0.05])
        # Measurement noise
        self.R_odom = np.diag([0.02, 0.02, 0.1, 0.05])
        self.R_marker = np.diag([0.01, 0.01, 0.01])

        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/marker_pose', self.marker_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)

        self.timer = self.create_timer(0.05, self.predict)

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def imu_callback(self, msg):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0 or dt > 1.0:
            return

        ax = msg.linear_acceleration.x
        omega = msg.angular_velocity.z

        self.state[3, 0] += ax * dt  # update v
        self.state[4, 0] = omega     # update omega

    def predict(self):
        dt = 0.05
        x, y, theta, v, omega = self.state.flatten()

        # State prediction
        x += v * np.cos(theta) * dt
        y += v * np.sin(theta) * dt
        theta += omega * dt

        theta = self.normalize_angle(theta)

        self.state = np.array([[x], [y], [theta], [v], [omega]])

        # Jacobian of motion model
        F = np.eye(5)
        F[0, 2] = -v * np.sin(theta) * dt
        F[0, 3] = np.cos(theta) * dt
        F[1, 2] = v * np.cos(theta) * dt
        F[1, 3] = np.sin(theta) * dt
        F[2, 4] = dt

        self.P = F @ self.P @ F.T + self.Q

        self.publish_pose()

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z

        z = np.array([[x], [y], [v], [omega]])
        H = np.zeros((4, 5))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 3] = 1  # v
        H[3, 4] = 1  # omega

        self.kalman_update(z, H, self.R_odom)

    def marker_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        theta = 2 * np.arctan2(qz, qw)

        z = np.array([[x], [y], [theta]])
        H = np.zeros((3, 5))
        H[0, 0] = 1  # x
        H[1, 1] = 1  # y
        H[2, 2] = 1  # theta

        self.kalman_update(z, H, self.R_marker)

    def kalman_update(self, z, H, R):
        y = z - H @ self.state
        if H.shape[0] >= 3:
            y[2, 0] = self.normalize_angle(y[2, 0])  # normalize theta residual
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state = self.state + K @ y
        self.state[2, 0] = self.normalize_angle(self.state[2, 0])  # normalize theta
        self.P = (np.eye(5) - K @ H) @ self.P

    def publish_pose(self):
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(self.state[0, 0])
        pose_msg.pose.position.y = float(self.state[1, 0])
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.z = math.sin(self.state[2, 0] / 2.0)
        pose_msg.pose.orientation.w = math.cos(self.state[2, 0] / 2.0)
        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
