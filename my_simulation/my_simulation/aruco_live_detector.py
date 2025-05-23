import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import transforms3d


class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.got_camera_info = False

        # Camera info subscriber
        self.create_subscription(CameraInfo, '/my_camera/camera/camera_info', self.camera_info_callback, 10)

        # Image subscriber
        self.create_subscription(Image, '/my_camera/camera/image_raw', self.image_callback, 10)

        # Publisher
        self.pose_pub = self.create_publisher(PoseStamped, '/marker_pose', 10)

        # ArUco setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.marker_length = 0.03  # meters

    def camera_info_callback(self, msg):
        if not self.got_camera_info:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.got_camera_info = True
            self.get_logger().info("Camera info received.")

    def image_callback(self, msg):
        if not self.got_camera_info:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

                for i in range(len(ids)):
                    pos = tvecs[i].ravel()
                    rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
                    quat = transforms3d.quaternions.mat2quat(rotation_matrix)
                    _, _, yaw = transforms3d.euler.mat2euler(rotation_matrix, axes='sxyz')

                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = 'my_camera_optical_frame'
                    pose_msg.pose.position.x = pos[0]
                    pose_msg.pose.position.y = pos[1]
                    pose_msg.pose.position.z = pos[2]
                    pose_msg.pose.orientation.x = quat[1]
                    pose_msg.pose.orientation.y = quat[2]
                    pose_msg.pose.orientation.z = quat[3]
                    pose_msg.pose.orientation.w = quat[0]
                    self.pose_pub.publish(pose_msg)

                    # Visualize
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_length)

            cv2.imshow("Aruco Detection", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Detection error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

