#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from tf_transformations import quaternion_matrix
import numpy as np
import math

class BayesFilterNode(Node):
    def __init__(self):
        super().__init__('bayes_filter_node')

        # State: [x, y, theta]
        self.state = np.zeros(3)
        self.velocity = np.zeros(2)
        self.P = np.eye(3) * 0.01

        self.process_noise = np.diag([0.05, 0.05, 0.01])
        self.measurement_noise = np.diag([0.1, 0.1, 0.05])

        self.last_time = None

        self.imu_sub = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.marker_sub = self.create_subscription(PoseStamped, '/marker_pose', self.marker_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)

    def imu_callback(self, msg: Imu):
        curr_time = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.last_time is None:
            self.last_time = curr_time
            return
        dt = curr_time - self.last_time
        self.last_time = curr_time

        # Extract quaternion orientation
        q = msg.orientation
        quat = [q.x, q.y, q.z, q.w]

        # Rotation matrix
        R = quaternion_matrix(quat)[:3, :3]

        # Raw acceleration in body frame
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        acc_body = np.array([ax, ay, az])

        # Rotate to world frame
        acc_world = R @ acc_body

        # Subtract gravity in world frame
        gravity_world = R @ np.array([0.0, 0.0, 9.81])
        acc_world -= gravity_world

        # Use x and y
        acc_xy = acc_world[:2]

        # Thresholding
        accel_thresh = 0.2
        acc_xy[np.abs(acc_xy) < accel_thresh] = 0.0


        # If stationary, slowly decay velocity
        if np.allclose(acc_xy, 0.0, atol=1e-3):
            self.velocity *= 0.9
        else:
            self.velocity += acc_xy * dt
            self.state[0:2] += self.velocity * dt

        # Yaw (theta) update from quaternion
        siny_cosp = 2 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1 - 2 * (q.y * q.y + q.z * q.z)
        theta = math.atan2(siny_cosp, cosy_cosp)
        self.state[2] = theta

        # Clamp small velocity drift
        self.velocity[np.abs(self.velocity) < 1e-3] = 0.0

        # Predict covariance
        self.P += self.process_noise * dt

        self.publish_pose()

    def marker_callback(self, msg: PoseStamped):
        z = np.array([msg.pose.position.x, msg.pose.position.y])

        # Kalman gain
        H = np.array([[1, 0, 0],
                      [0, 1, 0]])
        S = H @ self.P @ H.T + self.measurement_noise[:2, :2]
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state and covariance
        self.state[0:2] += K @ (z - H @ self.state)
        self.P = (np.eye(3) - K @ H) @ self.P

        self.publish_pose()

    def publish_pose(self):
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.position.x = self.state[0]
        pose_msg.pose.position.y = self.state[1]
        pose_msg.pose.position.z = 0.0

        # Convert yaw to quaternion
        theta = self.state[2]
        pose_msg.pose.orientation.w = math.cos(theta / 2)
        pose_msg.pose.orientation.z = math.sin(theta / 2)
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0

        self.pose_pub.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BayesFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


