import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
import numpy as np
import math


# the velocity is moving very slowly with this code and also, it is not moving backwards still
class SensorFusionNode(Node):
    def __init__(self):
        super().__init__('sensor_fusion_node')

        # Pose state: x, y, theta
        self.state = np.array([0.0, 0.0, 0.0])
        self.last_time = self.get_clock().now()

        # Last known motion
        self.vx = 0.0  # Velocity in x
        self.ax = 0.0  # Acceleration in x
        self.omega = 0.0  # Angular velocity (yaw rate)

        self.alpha = 0.9  # Trust in IMU; 1-alpha is trust in marker
        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)

        # Subscriptions
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(PoseStamped, '/marker_pose', self.marker_callback, 10)

        # Continuous update timer (Bayes prediction step)
        self.timer = self.create_timer(0.05, self.update_state)  # 20 Hz

        # Marker detection flag
        self.marker_detected = False

    def imu_callback(self, msg):
        self.ax = msg.linear_acceleration.x
        self.omega = msg.angular_velocity.z

    def update_state(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now

        if dt <= 0 or dt > 1:
            return

        x, y, theta = self.state

        # Update rotation
        dtheta = self.omega * dt
        theta += dtheta

        # Improved motion detection
        acceleration_threshold = 0.5  # m/s²
        omega_threshold = 0.1          # rad/s
        deceleration = 0.2  # Adjust as needed

        if abs(self.ax) > acceleration_threshold:
            self.vx += self.ax * dt
        else:
            # Apply deceleration toward zero
            if self.vx > 0:
                self.vx = max(0.0, self.vx - deceleration * dt)
            elif self.vx < 0:
                self.vx = min(0.0, self.vx + deceleration * dt)
        
        max_velocity = 0.5

        self.vx = np.clip(self.vx, -max_velocity, max_velocity)

        # Update position
        dx = self.vx * math.cos(theta) * dt
        dy = self.vx * math.sin(theta) * dt
        x += dx
        y += dy

        self.state = np.array([x, y, theta])
        self.publish_pose()


    def marker_callback(self, msg):
        measured_x = msg.pose.position.x
        measured_y = msg.pose.position.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        measured_theta = 2 * np.arctan2(qz, qw)

        measured_pose = np.array([measured_x, measured_y, measured_theta])

        if not self.marker_detected:
            self.alpha = 0.3 # trust the imu data less, and the marker more initially
            self.state = self.alpha * self.state + (1 - self.alpha) * measured_pose
            self.vx = 0.0
            self.marker_detected = True
        else:
            self.state = self.alpha * self.state + (1 - self.alpha) * measured_pose

        self.get_logger().info(f"[RESET] Corrected pose: {self.state}")

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
    node = SensorFusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
