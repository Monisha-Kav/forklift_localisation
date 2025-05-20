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

        self.odom_alpha = 0.8
        self.marker_alpha = 0.5

        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/marker_pose', self.marker_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)

        self.timer = self.create_timer(0.05, self.update_state)  # 20 Hz

    def imu_callback(self, msg):
        self.omega = msg.angular_velocity.z

        # Time delta
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        if dt <= 0 or dt > 1.0:
            return

        # Forward/backward motion logic
        raw_ax = msg.linear_acceleration.x
        theta = self.state[2]
        ax_world = raw_ax - 9.8 * math.sin(theta)  # gravity compensation

        if abs(ax_world) < 0.2:
            ax_world = 0.0

        if abs(self.omega) > 0.1 and abs(ax_world) < 0.05:
            self.vx = 0.0
        else:
            self.vx += ax_world * dt
            self.vx *= (1 - 0.05 * dt)
            self.vx = np.clip(self.vx, -0.5, 0.5)

        if abs(self.vx) < 1e-3:
            self.vx = 0.0

        # Rotation (yaw) from quaternion — from sensor_fusion_node.py
        q = msg.orientation
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)
        
        yaw_offset = -math.pi / 2
        theta += yaw_offset
        self.state[2] = math.atan2(math.sin(theta), math.cos(theta))

        self.last_time = current_time

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        measured = np.array([x, y])
        self.state[0:2] = self.odom_alpha * self.state[0:2] + (1 - self.odom_alpha) * measured

    def marker_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        measured = np.array([x, y])
        self.state[0:2] = self.marker_alpha * self.state[0:2] + (1 - self.marker_alpha) * measured

    def update_state(self):
        dt = 0.05
        prev_x, prev_y = self.state[0], self.state[1]

        dx = self.vx * math.cos(self.state[2]) * dt if abs(self.vx) > 1e-3 else 0.0
        dy = self.vx * math.sin(self.state[2]) * dt if abs(self.vx) > 1e-3 else 0.0

        new_x = prev_x + dx
        new_y = prev_y + dy

        smoothing_factor = 0.5
        self.state[0] = smoothing_factor * new_x + (1 - smoothing_factor) * prev_x
        self.state[1] = smoothing_factor * new_y + (1 - smoothing_factor) * prev_y

        # NOTE: theta is set from IMU quaternion already in imu_callback()

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

