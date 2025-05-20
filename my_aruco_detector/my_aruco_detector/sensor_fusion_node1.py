import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
import numpy as np
import math

class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        self.ax = 0.0
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

        # Raw acceleration in robot frame
        raw_ax = msg.linear_acceleration.x

        # Step 1: Remove gravity tilt effect using theta (already corrected by you)
        theta = self.state[2]
        ax_world = raw_ax - 9.8 * math.sin(theta)  # Remove gravity projected onto X

        # Step 2: Apply deadband to reject minor noise
        if abs(ax_world) < 0.2:
            ax_world = 0.0

        # Step 3: Integrate acceleration to velocity (clip too)
        self.vx += ax_world * dt
        
        friction = 0.05
        self.vx *= (1 - friction * dt)
    
        self.vx = np.clip(self.vx, -0.5, 0.5)  # limit max speed
        
        if ax_world == 0.0 and abs(self.vx) < 0.02:
            self.vx = 0.0

        # Step 4: Compute new position
        dx = self.vx * math.cos(theta) * dt
        dy = self.vx * math.sin(theta) * dt
        dtheta = self.omega * dt

        self.state += np.array([dx, dy, dtheta])

        self.get_logger().info(f"[IMU] ax_world={ax_world:.3f} vx={self.vx:.3f} -> state: {self.state}")


    def gyro_callback(self, msg):
        self.omega = msg.angular_velocity.z

    def update_state(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0 or dt > 1.0:
            return

        # Apply deadband to filter noise
        ax = self.ax
        if abs(ax) < 0.02:
            ax = 0.0

        # Integrate velocity
        self.vx += ax * dt

        # Apply velocity deadband and clipping
        if abs(self.vx) < 0.01:
            self.vx = 0.0
            
        self.vx = np.clip(self.vx, -0.5, 0.5)

        # Update state
        x, y, theta = self.state
        x += self.vx * math.cos(theta) * dt
        y += self.vx * math.sin(theta) * dt
        theta += self.omega * dt

        self.state = np.array([x, y, theta])

        self.publish_pose()
        self.get_logger().info(f"[IMU] Predicted: {self.state}")

    def marker_callback(self, msg):
        measured_x = msg.pose.position.x
        measured_y = msg.pose.position.y

        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        measured_theta = 2.0 * np.arctan2(qz, qw)

        measured_state = np.array([measured_x, measured_y, measured_theta])

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

