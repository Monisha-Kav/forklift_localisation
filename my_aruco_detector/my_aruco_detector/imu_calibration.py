import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu, Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import cv2.aruco as aruco

class IMUCalibrator:
    def __init__(self, alpha=0.05):
        self.bias_accel = np.zeros(3)
        self.bias_gyro = np.zeros(3)
        self.alpha = alpha

    def update_bias(self, accel, gyro):
        self.bias_accel = (1 - self.alpha) * self.bias_accel + self.alpha * accel
        self.bias_gyro = (1 - self.alpha) * self.bias_gyro + self.alpha * gyro

    def calibrate(self, accel, gyro):
        return accel - self.bias_accel, gyro - self.bias_gyro

class TurtleBot3EKFLocalizer(Node):
    def __init__(self):
        super().__init__('turtlebot3_ekf_localizer')

        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)
        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Image, '/camera/image_raw', self.image_callback, 10)

        self.bridge = CvBridge()
        self.calibrator = IMUCalibrator()
        self.state = np.zeros(3)  # x, y, theta
        self.P = np.eye(3)
        self.Q = np.diag([0.05, 0.05, 0.01])
        self.R_aruco = np.diag([0.02, 0.02, 0.01])

        self.last_time = self.get_clock().now()
        self.ax = 0.0
        self.omega = 0.0

        self.camera_matrix = np.eye(3)
        self.dist_coeffs = np.zeros(5)
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.detector_params = aruco.DetectorParameters_create()

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
        if dt <= 0 or dt > 1:
            return
        self.last_time = now
        theta = self.state[2]
        v = self.ax * dt
        dx = v * np.cos(theta) * dt
        dy = v * np.sin(theta) * dt
        dtheta = self.omega * dt
        self.state += np.array([dx, dy, dtheta])
        F = np.array([
            [1, 0, -v * np.sin(theta) * dt],
            [0, 1,  v * np.cos(theta) * dt],
            [0, 0, 1]
        ])
        self.P = F @ self.P @ F.T + self.Q
        self.publish_pose()

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco.detectMarkers(gray, self.aruco_dict, parameters=self.detector_params)
        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.05, self.camera_matrix, self.dist_coeffs)
            z = np.array([tvecs[0][0][0], tvecs[0][0][1], 0.0])
            H = np.eye(3)
            y = z - self.state
            S = H @ self.P @ H.T + self.R_aruco
            K = self.P @ H.T @ np.linalg.inv(S)
            self.state += K @ y
            self.P = (np.eye(3) - K @ H) @ self.P
            self.get_logger().info(f"[ARUCO] Correction → x: {self.state[0]:.2f}, y: {self.state[1]:.2f}")

    def publish_pose(self):
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'map'
        pose.pose.position.x = float(self.state[0])
        pose.pose.position.y = float(self.state[1])
        pose.pose.position.z = 0.0
        pose.pose.orientation.z = np.sin(self.state[2] / 2)
        pose.pose.orientation.w = np.cos(self.state[2] / 2)
        self.pose_pub.publish(pose)

def main(args=None):
    rclpy.init(args=args)
    node = TurtleBot3EKFLocalizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
