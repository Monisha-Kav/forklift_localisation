import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import transforms3d

class MarkerDetector(Node):
    def __init__(self):
        super().__init__('marker_detector')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/image_raw/compressed',
            self.listener_callback,
            10
        )
        self.pose_pub = self.create_publisher(PoseStamped, '/marker_pose', 10)

        # Camera parameters
        self.camera_matrix = np.array([
            [506.7045, 0, 317.0278],
            [0, 506.7707, 231.4871],
            [0, 0, 1]
        ], dtype=np.float32)
        self.dist_coeffs = np.array([[0.182278215942289], [-0.306125275008631], [0], [0], [0]], dtype=np.float32)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.parameters = cv2.aruco.DetectorParameters_create()
        self.marker_length = 0.03  # in meters

    def listener_callback(self, msg):
        try:
            # Decode the compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect markers
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
                for i in range(len(ids)):
                    # Convert to position and orientation
                    position = tvecs[i].ravel()
                    rotation_matrix, _ = cv2.Rodrigues(rvecs[i])
                    quaternion = transforms3d.quaternions.mat2quat(rotation_matrix)

                    # Calculate yaw (theta)
                    _, _, yaw = transforms3d.euler.mat2euler(rotation_matrix, axes='sxyz')

                    # Publish pose
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = 'base_link'
                    pose_msg.pose.position.x = position[0]
                    pose_msg.pose.position.y = position[1]
                    pose_msg.pose.orientation.z = np.sin(yaw / 2)
                    pose_msg.pose.orientation.w = np.cos(yaw / 2)
                    self.pose_pub.publish(pose_msg)

                    # Visualization (optional)
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], self.marker_length)
            cv2.imshow("ArUco Marker Detection", frame)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    marker_detector = MarkerDetector()
    rclpy.spin(marker_detector)
    marker_detector.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
