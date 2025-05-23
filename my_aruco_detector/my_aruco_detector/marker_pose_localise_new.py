import rclpy
from rclpy.node import Node
import cv2
import numpy as np
import yaml
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
import tf_transformations
import tf2_ros
import geometry_msgs.msg


class ArucoLocalizer(Node):
    def __init__(self):
        super().__init__('aruco_localizer')

        # Declare parameters
        self.declare_parameter('marker_size', 0.1)
        self.declare_parameter('aruco_marker_map', 'aruco_marker_map.yaml')
        self.declare_parameter('image_topic', '/camera/image_raw')
        self.declare_parameter('camera_info_topic', '/camera/camera_info')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')

        self.marker_size = self.get_parameter('marker_size').value
        self.aruco_marker_map_path = self.get_parameter('aruco_marker_map').value
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.base_frame = self.get_parameter('base_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value

        # Load the marker map
        with open(self.aruco_marker_map_path, 'r') as f:
            raw_map = yaml.safe_load(f)

        self.known_markers = {}
        self.goal_markers = {}

        for marker_id, data in raw_map.items():
            if 'position' in data:
                pos = data['position']
                yaw = data.get('yaw', 0.0)
                self.known_markers[int(marker_id)] = {
                    'position': [pos[0], pos[1], 0.0],
                    'orientation': [0.0, 0.0, yaw]
                }
            elif 'goal' in data:
                goal = data['goal']
                self.goal_markers[int(marker_id)] = {
                    'goal': [goal[0], goal[1], 0.0]
                }

        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None
        self.camera_info_received = False

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.info_sub = self.create_subscription(CameraInfo, self.camera_info_topic, self.info_callback, 10)

        self.pose_pub = self.create_publisher(PoseStamped, '/estimate', 10)
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_position', 10)

        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        self.get_logger().info("ArucoLocalizer initialized.")

    def info_callback(self, msg):
        if not self.camera_info_received:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.camera_info_received = True
            self.get_logger().info("Camera info received.")

    def image_callback(self, msg):
        if not self.camera_info_received:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters_create()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                    [corners[i]], self.marker_size, self.camera_matrix, self.dist_coeffs)

                rvec = rvecs[0][0]
                tvec = tvecs[0][0]
                R_ct, _ = cv2.Rodrigues(rvec)
                T_ct = np.eye(4)
                T_ct[:3, :3] = R_ct
                T_ct[:3, 3] = tvec

                if marker_id in self.known_markers:
                    marker = self.known_markers[marker_id]
                    R_wm = tf_transformations.euler_matrix(*marker['orientation'])[:3, :3]
                    T_wm = np.eye(4)
                    T_wm[:3, :3] = R_wm
                    T_wm[:3, 3] = marker['position']

                    T_mc = T_ct
                    T_cm = np.linalg.inv(T_mc)
                    T_wc = T_wm @ T_cm

                    position = T_wc[:3, 3]
                    quat = tf_transformations.quaternion_from_matrix(T_wc)

                    # Publish pose
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = msg.header.stamp
                    pose_msg.header.frame_id = 'map'
                    pose_msg.pose.position.x = position[0]
                    pose_msg.pose.position.y = position[1]
                    pose_msg.pose.position.z = position[2]
                    pose_msg.pose.orientation.x = quat[0]
                    pose_msg.pose.orientation.y = quat[1]
                    pose_msg.pose.orientation.z = quat[2]
                    pose_msg.pose.orientation.w = quat[3]
                    self.pose_pub.publish(pose_msg)

                    # Broadcast TF
                    t = geometry_msgs.msg.TransformStamped()
                    t.header.stamp = msg.header.stamp
                    t.header.frame_id = 'map'
                    t.child_frame_id = self.camera_frame
                    t.transform.translation.x = position[0]
                    t.transform.translation.y = position[1]
                    t.transform.translation.z = position[2]
                    t.transform.rotation.x = quat[0]
                    t.transform.rotation.y = quat[1]
                    t.transform.rotation.z = quat[2]
                    t.transform.rotation.w = quat[3]
                    self.tf_broadcaster.sendTransform(t)

                    break  # use first known marker

                elif marker_id in self.goal_markers:
                    goal_pos = self.goal_markers[marker_id]['goal']
                    goal_msg = PoseStamped()
                    goal_msg.header.stamp = msg.header.stamp
                    goal_msg.header.frame_id = 'map'
                    goal_msg.pose.position.x = goal_pos[0]
                    goal_msg.pose.position.y = goal_pos[1]
                    goal_msg.pose.position.z = goal_pos[2]
                    goal_msg.pose.orientation.w = 1.0
                    self.goal_pub.publish(goal_msg)
                    self.get_logger().info(f"Goal marker {marker_id} detected. Goal: {goal_pos[:2]}")


def main(args=None):
    rclpy.init(args=args)
    node = ArucoLocalizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

