import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
import math

class EKFSensorFusionNode(Node):
    def __init__(self):
        super().__init__('ekf_sensor_fusion_node')

        # State: x, y, theta, vx, omega
        self.x = np.zeros((5, 1))

        # Covariance matrix
        self.P = np.eye(5) * 0.1

        # Process noise
        self.Q = np.diag([0.02, 0.02, 0.01, 0.1, 0.1])

        # Measurement noise (odom and marker)
        self.R_odom = np.diag([0.1, 0.1, 0.05])
        self.R_marker = np.diag([0.05, 0.05, 0.03])

        self.last_time = self.get_clock().now()

        # Subscriptions
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/marker_pose', self.marker_callback, 10)

        # Publisher
        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)

        # Timer for prediction update
        self.timer = self.create_timer(0.05, self.predict)  # 20 Hz

    def imu_callback(self, msg):
        self.x[4, 0] = msg.angular_velocity.z

        ax = msg.linear_acceleration.x
        theta = self.x[2, 0]
        ax_world = ax - 9.8 * math.sin(theta)

        if abs(ax_world) < 0.2:
            ax_world = 0.0

        self.x[3, 0] += ax_world * 0.05  # update vx
        self.x[3, 0] = np.clip(self.x[3, 0], -1.0, 1.0)

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        qz = msg.pose.pose.orientation.z
        qw = msg.pose.pose.orientation.w
        theta = 2 * np.arctan2(qz, qw)

        z = np.array([[x], [y], [theta]])
        self.ekf_update(z, self.R_odom)

    def marker_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        theta = 2 * np.arctan2(qz, qw)

        z = np.array([[x], [y], [theta]])
        self.ekf_update(z, self.R_marker)

    def predict(self):
        dt = 0.05
        theta = self.x[2, 0]
        vx = self.x[3, 0]
        omega = self.x[4, 0]

        # State prediction
        self.x[0, 0] += vx * math.cos(theta) * dt
        self.x[1, 0] += vx * math.sin(theta) * dt
        self.x[2, 0] += omega * dt

        # Normalize angle
        self.x[2, 0] = (self.x[2, 0] + np.pi) % (2 * np.pi) - np.pi

        # Jacobian of motion model
        F = np.eye(5)
        F[0, 2] = -vx * math.sin(theta) * dt
        F[0, 3] = math.cos(theta) * dt
        F[1, 2] = vx * math.cos(theta) * dt
        F[1, 3] = math.sin(theta) * dt
        F[2, 4] = dt

        self.P = F @ self.P @ F.T + self.Q

        self.publish_pose()

    def ekf_update(self, z, R):
        # Measurement function: position and orientation only
        H = np.zeros((3, 5))
        H[0, 0] = 1
        H[1, 1] = 1
        H[2, 2] = 1

        z_pred = H @ self.x
        y = z - z_pred

        # Normalize angle
        y[2, 0] = (y[2, 0] + np.pi) % (2 * np.pi) - np.pi

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (np.eye(5) - K @ H) @ self.P

    def publish_pose(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(self.x[0, 0])
        pose.pose.position.y = float(self.x[1, 0])
        pose.pose.position.z = 0.0

        yaw = self.x[2, 0]
        pose.pose.orientation.z = math.sin(yaw / 2)
        pose.pose.orientation.w = math.cos(yaw / 2)

        self.pose_pub.publish(pose)


def main(args=None):
    rclpy.init(args=args)
    node = EKFSensorFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
