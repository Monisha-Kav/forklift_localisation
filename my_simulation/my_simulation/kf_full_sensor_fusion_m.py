# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Imu
# from nav_msgs.msg import Odometry
# from geometry_msgs.msg import PoseStamped
# import numpy as np
# import math
# from std_msgs.msg import Bool

# class KalmanFilterNode(Node):
#     def __init__(self):
#         super().__init__('kalman_sensor_fusion_node')

#         # [x, y, theta, v, omega]
#         self.state = np.zeros((5, 1))
#         self.P = np.eye(5) * 0.1

#         self.last_time = self.get_clock().now()

#         self.Q = np.diag([0.01, 0.01, 0.01, 0.05, 0.05])
#         self.R_odom = np.diag([0.02, 0.02, 0.1, 0.05, 0.05])
#         self.R_marker = np.diag([0.05, 0.05, 0.05])  # Less trust in visual

#         self.create_subscription(Imu, '/imu', self.imu_callback, 10)
#         self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
#         self.create_subscription(PoseStamped, '/marker_pose', self.marker_callback, 10)
#         self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)

#         self.timer = self.create_timer(0.05, self.predict)

#         # Marker filtering
#         self.last_marker_pose = None
#         self.alpha = 0.6
#         self.last_marker_update_time = self.get_clock().now()
#         self.marker_update_interval = rclpy.duration.Duration(seconds=0.2)

#         self.marker_visible = False

#         self.log_file = open('/tmp/mahalanobis_log.txt', 'w')  # Or choose your own path
#         self.log_file.write("timestamp,mahalanobis_distance,accepted\n")

#         self.marker_buffer = []
#         self.marker_reset_done = False
#         self.marker_first_seen_time = None
        
#         self.calibrated_pub = self.create_publisher(Bool, '/calibration_status', 10)




#     def normalize_angle(self, angle):
#         return (angle + np.pi) % (2 * np.pi) - np.pi

#     def imu_callback(self, msg):
#         now = self.get_clock().now()
#         dt = (now - self.last_time).nanoseconds * 1e-9
#         self.last_time = now
#         if dt <= 0 or dt > 1.0:
#             return

#         ax = msg.linear_acceleration.x
#         omega = msg.angular_velocity.z

#         self.state[3, 0] += ax * dt
#         self.state[4, 0] = omega

#     def predict(self):
#         dt = 0.05
#         x, y, theta, v, omega = self.state.flatten()

#         # Motion model
#         x += v * np.cos(theta) * dt
#         y += v * np.sin(theta) * dt
#         theta += omega * dt
#         theta = self.normalize_angle(theta)

#         # self.state = np.array([[x], [y], [theta], [v], [omega]])
#         self.state = np.array([[v], [omega]])


#         # Jacobian
#         F = np.eye(5)
#         F[0, 2] = -v * np.sin(theta) * dt
#         F[0, 3] = np.cos(theta) * dt
#         F[1, 2] = v * np.cos(theta) * dt
#         F[1, 3] = np.sin(theta) * dt
#         F[2, 4] = dt

#         self.P = F @ self.P @ F.T + self.Q
#         self.publish_pose()

#     def odom_callback(self, msg):
#         # x = msg.pose.pose.position.x
#         # y = msg.pose.pose.position.y
#         qz = msg.pose.pose.orientation.z
#         qw = msg.pose.pose.orientation.w
#         theta = 2 * np.arctan2(qz, qw)
#         v = msg.twist.twist.linear.x
#         omega = msg.twist.twist.angular.z

#         # z = np.array([[x], [y], [theta], [v], [omega]])
#         # H = np.eye(5)
#         # self.kalman_update(z, H, self.R_odom)

#         z = np.array([[v], [omega]])
#         H = np.eye(5)
#         self.kalman_update(z, H, self.R_odom)




import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool
import numpy as np
import math
import os

class KalmanFilterNode(Node):
    def __init__(self):
        super().__init__('kalman_sensor_fusion_node')

        # [x, y, theta, v, omega]
        self.state = np.zeros((5, 1))
        self.P = np.eye(5) * 0.1

        self.last_time = self.get_clock().now()

        self.Q = np.diag([0.01, 0.01, 0.01, 0.05, 0.05])
        self.R_odom = np.diag([0.02, 0.02, 0.1, 0.05, 0.05])
        self.R_marker = np.diag([0.05, 0.05, 0.05])  # Less trust in visual

        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/marker_pose', self.marker_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)
        self.calibrated_pub = self.create_publisher(Bool, '/calibration_status', 10)

        self.timer = self.create_timer(0.05, self.predict)

        self.marker_buffer = []
        self.marker_reset_done = False
        self.marker_first_seen_time = None
        self.last_marker_update_time = self.get_clock().now()
        self.marker_update_interval = rclpy.duration.Duration(seconds=0.2)

        # Create log file
        log_path = '/tmp/mahalanobis_log.txt'
        self.log_file = open(log_path, 'w')
        self.log_file.write("timestamp,mahalanobis_distance,accepted\n")

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

        self.state[3, 0] += ax * dt
        self.state[4, 0] = omega

    def predict(self):
        dt = 0.05
        x, y, theta, v, omega = self.state.flatten()

        # Motion model
        x += v * math.cos(theta) * dt
        y += v * math.sin(theta) * dt
        theta += omega * dt
        theta = self.normalize_angle(theta)

        self.state[0, 0] = x
        self.state[1, 0] = y
        self.state[2, 0] = theta

        # Jacobian
        F = np.eye(5)
        F[0, 2] = -v * math.sin(theta) * dt
        F[0, 3] = math.cos(theta) * dt
        F[1, 2] = v * math.cos(theta) * dt
        F[1, 3] = math.sin(theta) * dt
        F[2, 4] = dt

        self.P = F @ self.P @ F.T + self.Q
        self.publish_pose()

    def odom_callback(self, msg):
        v = msg.twist.twist.linear.x
        omega = msg.twist.twist.angular.z

        z = np.array([[v], [omega]])
        H = np.zeros((2, 5))
        H[0, 3] = 1.0
        H[1, 4] = 1.0
        R = np.diag([0.05, 0.05])
        self.kalman_update(z, H, R)

    def marker_callback(self, msg):
        now = self.get_clock().now()
        if (now - self.last_marker_update_time) < self.marker_update_interval:
            return
        self.last_marker_update_time = now

        x = msg.pose.position.x
        y = msg.pose.position.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        theta = 2 * math.atan2(qz, qw)

        if self.marker_first_seen_time is None:
            self.marker_first_seen_time = now
            self.marker_buffer = []
            self.get_logger().info("Started buffering marker data...")

        self.marker_buffer.append((x, y, theta))
        time_elapsed = (now - self.marker_first_seen_time).nanoseconds * 1e-9
        if time_elapsed < 2.0:
            self.get_logger().info(f"Buffering marker... ({time_elapsed:.1f}s)")
            return

        # Only gets here AFTER 2 seconds
        avg_x = np.mean([p[0] for p in self.marker_buffer])
        avg_y = np.mean([p[1] for p in self.marker_buffer])
        thetas = [p[2] for p in self.marker_buffer]
        avg_theta = math.atan2(np.mean([math.sin(t) for t in thetas]),
                               np.mean([math.cos(t) for t in thetas]))

        self.state[0, 0] = avg_x
        self.state[1, 0] = avg_y
        self.state[2, 0] = self.normalize_angle(avg_theta)
        self.state[3, 0] = 0.0
        self.state[4, 0] = 0.0

        self.get_logger().info(f"Pose RESET using marker avg: x={avg_x:.2f}, y={avg_y:.2f}, theta={avg_theta:.2f}")

        self.marker_reset_done = True
        self.marker_first_seen_time = None
        self.marker_buffer = []

        # Publish calibration success
        msg = Bool()
        msg.data = True
        self.calibrated_pub.publish(msg)

    def kalman_update(self, z, H, R):
        y = z - H @ self.state
        if H.shape[0] >= 3:
            y[2, 0] = self.normalize_angle(y[2, 0])
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.state[2, 0] = self.normalize_angle(self.state[2, 0])
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

        # self.get_logger().info(f"[KF] Publishing fused_pose: x={self.state[0,0]:.2f}, y={self.state[1,0]:.2f}")

    def destroy_node(self):
        if hasattr(self, 'log_file'):
            self.log_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()