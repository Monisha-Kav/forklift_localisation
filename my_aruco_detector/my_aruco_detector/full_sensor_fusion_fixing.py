import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
import math

class FullSensorFusionNode(Node):
    def __init__(self):
        super().__init__('full_sensor_fusion_node')

        self.state = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.vx = 0.0
        self.omega = 0.0

        self.last_time = self.get_clock().now()

        self.odom_alpha = 0.6
        self.marker_alpha = 0.5
        self.imu_marker_alpha = 0.9  # theta: trust IMU more than ArUco

        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/marker_pose', self.marker_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)

        self.timer = self.create_timer(0.05, self.update_state)  # 20 Hz

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def imu_callback(self, msg):
        self.omega = msg.angular_velocity.z

        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        if dt <= 0 or dt > 1.0:
            return

        raw_ax = msg.linear_acceleration.x
        theta = self.state[2]
        ax_world = raw_ax - 9.8 * math.sin(theta)

        if abs(ax_world) < 0.2:
            ax_world = 0.0

        self.vx += ax_world * dt
        self.vx *= (1 - 0.05 * dt)
        self.vx = np.clip(self.vx, -0.5, 0.5)

        self.last_time = current_time

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        # Only use x and y (not theta)
        measured = np.array([x, y])
        self.state[0:2] = self.odom_alpha * self.state[0:2] + (1 - self.odom_alpha) * measured

    def marker_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        marker_theta = 2 * np.arctan2(qz, qw)

        # Position correction (same as before)
        measured_xy = np.array([x, y])
        self.state[0:2] = self.marker_alpha * self.state[0:2] + (1 - self.marker_alpha) * measured_xy

        # Orientation correction like sensor_fusion_node_drift.py
        self.state[2] = self.imu_marker_alpha * self.state[2] + (1 - self.imu_marker_alpha) * marker_theta
        self.state[2] = self.normalize_angle(self.state[2])

    def update_state(self):
        dt = 0.05
        dx = self.vx * math.cos(self.state[2]) * dt
        dy = self.vx * math.sin(self.state[2]) * dt
        dtheta = self.omega * dt

        self.state[0] += dx
        self.state[1] += dy
        self.state[2] += dtheta
        self.state[2] = self.normalize_angle(self.state[2])

        self.publish_pose()

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
    node = FullSensorFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

