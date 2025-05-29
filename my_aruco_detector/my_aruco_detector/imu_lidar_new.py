import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, LaserScan
from geometry_msgs.msg import PoseStamped
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

class IMUCalibrator:
    def __init__(self, alpha=0.01):
        self.bias_accel = np.zeros(3)
        self.bias_gyro = np.zeros(3)
        self.alpha = alpha

    def update_bias(self, accel, gyro):
        self.bias_accel = (1 - self.alpha) * self.bias_accel + self.alpha * accel
        self.bias_gyro = (1 - self.alpha) * self.bias_gyro + self.alpha * gyro

    def calibrate(self, accel, gyro):
        return accel - self.bias_accel, gyro - self.bias_gyro

class TurtleBotEKF(Node):
    def __init__(self):
        super().__init__('imu_lidar_ekf')
        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        lidar_qos = QoSProfile(depth=10)
        lidar_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        
        self.create_subscription(LaserScan, '/scan', self.lidar_callback, qos_profile=lidar_qos)

        self.calibrator = IMUCalibrator()

        self.state = np.zeros(3)  # [x, y, theta]
        self.v = 0.0
        self.P = np.eye(3) * 0.01

        self.Q = np.diag([0.01, 0.01, 0.005])
        self.R_lidar = np.diag([0.05, 0.05, 0.01])

        self.ax = 0.0
        self.omega = 0.0
        self.last_time = self.get_clock().now()

        self.create_timer(0.05, self.prediction_step)

    def imu_callback(self, msg):
        accel = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z])
        gyro = np.array([msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
        self.calibrator.update_bias(accel, gyro)
        accel_cal, gyro_cal = self.calibrator.calibrate(accel, gyro)
        self.ax = accel_cal[0]
        self.omega = gyro_cal[2]

    def prediction_step(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        if dt <= 0.0 or dt > 1.0:
            return
        self.last_time = now

        theta = self.state[2]
        self.v += self.ax * dt

        # adding velocity damping
        self.v *= 0.98

        dx = self.v * np.cos(theta) * dt
        dy = self.v * np.sin(theta) * dt
        dtheta = self.omega * dt

        self.state += np.array([dx, dy, dtheta])

        F = np.array([
            [1, 0, -self.v * np.sin(theta) * dt],
            [0, 1,  self.v * np.cos(theta) * dt],
            [0, 0, 1]
        ])
        self.P = F @ self.P @ F.T + self.Q

        self.state[0] = np.clip(self.state[0], 0.0, 2.0)
        self.state[1] = np.clip(self.state[1], 0.0, 2.0)

        self.publish_pose()

    def lidar_callback(self, msg):
        ranges = np.array(msg.ranges)
        angle_min = msg.angle_min
        angle_max = msg.angle_max
        angle_inc = msg.angle_increment
        num_beams = len(ranges)

        theta = self.state[2]
        theta = (theta + 2*np.pi) % (2*np.pi)  # normalize to [0, 2pi]

        # Get index of beam closest to robot heading
        angle = angle_min
        idx = int((theta - angle_min) / angle_inc)
        if idx < 0 or idx >= num_beams:
            return

        range_at_theta = ranges[idx]
        if np.isnan(range_at_theta) or range_at_theta < 0.1 or range_at_theta > 3.0:
            return  # Invalid reading

        # Based on heading, choose which wall we're facing and correct accordingly
        z = self.state.copy()
        epsilon = 0.6  # how close theta must be to face a wall
        corrected = False

        if abs(theta - 0.0) < epsilon or abs(theta - 2*np.pi) < epsilon:
            # Facing +x wall at x=2.0
            z[0] = 2.0 - range_at_theta
            corrected = True
        elif abs(theta - np.pi/2) < epsilon:
            # Facing +y wall at y=2.0
            z[1] = 2.0 - range_at_theta
            corrected = True
        elif abs(theta - np.pi) < epsilon:
            # Facing -x wall at x=0.0
            z[0] = 0.0 + range_at_theta
            corrected = True
        elif abs(theta - 3*np.pi/2) < epsilon:
            # Facing -y wall at y=0.0
            z[1] = 0.0 + range_at_theta
            corrected = True

        if corrected:
            H = np.eye(3)
            y = z - self.state
            S = H @ self.P @ H.T + self.R_lidar
            K = self.P @ H.T @ np.linalg.inv(S)
            self.state += K @ y
            self.P = (np.eye(3) - K @ H) @ self.P
        if not corrected:
            self.get_logger().info(f"[LIDAR] Skipped correction — theta={theta:.2f} not near wall direction")


            self.get_logger().info(f"[LIDAR] Got scan at theta={theta:.2f} rad, range={range_at_theta:.2f}")
            self.get_logger().info(f"[LIDAR] Correction → x: {self.state[0]:.2f}, y: {self.state[1]:.2f}")

    def publish_pose(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "map"
        pose.pose.position.x = float(self.state[0])
        pose.pose.position.y = float(self.state[1])
        pose.pose.position.z = 0.0

        theta = self.state[2]
        pose.pose.orientation.z = np.sin(theta / 2.0)
        pose.pose.orientation.w = np.cos(theta / 2.0)
        self.pose_pub.publish(pose)

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBotEKF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()

