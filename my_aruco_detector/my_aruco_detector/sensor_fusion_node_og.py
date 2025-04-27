import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
import numpy as np

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')
        
        # Subscriptions
        self.create_subscription(Imu, '/imu/data', self.imu_callback, 10)
        self.create_subscription(PoseStamped, '/marker_pose', self.marker_callback, 10)

        # Internal state
        self.state = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.last_time = self.get_clock().now()
        self.marker_seen = False

        self.alpha = 0.8  # Trust in IMU, 1-alpha trust in ArUco

    def imu_callback(self, msg):
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds * 1e-9
        self.last_time = current_time

        if dt <= 0 or dt > 1:  # Skip if time diff is too large
            return

        # IMU data
        ax = msg.linear_acceleration.x
        omega = msg.angular_velocity.z

        # Predict position based on last pose
        theta = self.state[2]
        v = ax * dt  # Rough estimation of velocity
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = omega * dt

        predicted_state = self.state + np.array([dx, dy, dtheta])
        self.state = predicted_state

        self.get_logger().info(f"[IMU] Predicted: {self.state}")

    def marker_callback(self, msg):
        # Marker pose is relative to robot, but assume we know where the marker is in world
        # OR use this as a correction to global pose

        # Simple correction (pretending marker position is robot's absolute pose)
        measured_x = msg.pose.position.x
        measured_y = msg.pose.position.y

        # Extract yaw from quaternion
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        measured_theta = 2 * np.arctan2(qz, qw)

        measured_pose = np.array([measured_x, measured_y, measured_theta])

        # Fuse using Bayes-style weighted average
        self.state = self.alpha * self.state + (1 - self.alpha) * measured_pose
        self.marker_seen = True

        self.get_logger().info(f"[Fusion] Updated Pose: {self.state}")

def main(args=None):
    rclpy.init(args=args)
    node = SensorFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()