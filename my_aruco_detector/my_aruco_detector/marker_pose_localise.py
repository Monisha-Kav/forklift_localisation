import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import CompressedImage
import cv2
import numpy as np
import tf_transformations
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import yaml
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
        self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
        self.br = TransformBroadcaster(self)

        # Camera parameters
        self.camera_matrix = np.array([
            [920.7894, 0, 589.1690],
            [0, 919.1768, 541.4353],
            [0, 0, 1.0000]
        ], dtype=np.float32)
        self.dist_coeffs = np.array([[0.205245304646941], [-0.332544106955044], [0], [0], [0]], dtype=np.float32)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.parameters = cv2.aruco.DetectorParameters()

        # Load marker positions and goals
        yaml_path = '/home/monisha/turtlebot3_ws/src/my_aruco_detector/my_aruco_detector/aruco_marker_map.yaml'
        with open(yaml_path, 'r') as f:
            marker_map = yaml.safe_load(f)

        self.marker_world_positions = {}
        self.goal_markers = {}
        for marker_id, data in marker_map.items():
            if 'position' in data and 'orientation' in data:
                self.marker_world_positions[int(marker_id)] = {
                    'position': data['position'],
                    'orientation': data['orientation']
                }

            if 'goal' in data:
                self.goal_markers[int(marker_id)] = data['goal']

        self.marker_length = 0.03  # marker side length in meters

    def listener_callback(self, msg):
        try:
            # Decode the compressed image
            np_arr = np.frombuffer(msg.data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            # results = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
            # corners, ids = results[0], results[1]
            corners, ids, rejectedImgPoints, *_ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)



            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)
                for i in range(len(ids)):
                    marker_id = int(ids[i][0])

                    # Set axis color and label color
                    is_goal = marker_id in self.goal_markers
                    axis_length = 0.02  # 2 cm
                    color = (0, 255, 0) if is_goal else (255, 0, 0)  # Green for goal, Blue for regular

                    # # Draw the axis (OpenCV uses internal color logic, draw with default for all)
                    # cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], axis_length)

                    # Draw the marker ID
                    c = corners[i][0].mean(axis=0).astype(int)
                    label = f"Goal ID: {marker_id}" if is_goal else f"ID: {marker_id}"
                    cv2.putText(frame, label, (c[0], c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    # First, if it is a goal marker
                    if marker_id in self.goal_markers:
                        goal_x, goal_y = self.goal_markers[marker_id]
                        goal_msg = PoseStamped()
                        goal_msg.header.stamp = self.get_clock().now().to_msg()
                        goal_msg.header.frame_id = 'map'
                        goal_msg.pose.position.x = goal_x
                        goal_msg.pose.position.y = goal_y
                        goal_msg.pose.position.z = 0.0
                        goal_msg.pose.orientation.w = 1.0  # no orientation for goal
                        self.goal_pub.publish(goal_msg)
                        continue  # DO NOT reset pose with goal markers

                    if marker_id not in self.marker_world_positions:
                        continue

                    # Estimate pose of the marker w.r.t. camera frame
                    marker_half = self.marker_length / 2.0
                    obj_points = np.array([
                        [-marker_half,  marker_half, 0],
                        [ marker_half,  marker_half, 0],
                        [ marker_half, -marker_half, 0],
                        [-marker_half, -marker_half, 0]
                    ], dtype=np.float32)

                    success, rvec, tvec = cv2.solvePnP(
                        obj_points,
                        corners[0],
                        self.camera_matrix,   # come back to this line, might cause an error due to mismatch between this name btw sim and hardware
                        self.dist_coeffs,
                        flags=cv2.SOLVEPNP_IPPE_SQUARE
                    )

                    self.get_logger().info(f"tvec: {tvec.flatten()} rvec: {rvec.flatten()}")

                    if not success:
                        self.get_logger().warn("Pose estimation failed.")
                        continue

                    # Pose estimation
                    R_cm, _ = cv2.Rodrigues(rvec)
                    T_cm = np.eye(4)
                    T_cm[:3, :3] = R_cm
                    T_cm[:3, 3] = tvec.flatten()

                    # OpenCV to Gazebo transform
                    T_cv_to_gazebo = np.array([
                        [ 0,  0,  -1, 0],
                        [ 1,  0,  0, 0],
                        [ 0, -1,  0, 0],
                        [ 0,  0,  0, 1]
                    ])

                    # Get marker's world pose
                    # marker_pos, marker_quat = self.marker_world_positions[marker_id]
                    marker_pos = self.marker_world_positions[marker_id]['position']
                    marker_quat = self.marker_world_positions[marker_id]['orientation']

                    R_wm = transforms3d.quaternions.quat2mat(marker_quat)
                    T_wm = np.eye(4)
                    T_wm[:3, :3] = R_wm
                    T_wm[:3, 3] = marker_pos

                    # Camera pose in world frame
                    T_wc = T_wm @ np.linalg.inv(T_cv_to_gazebo @ T_cm @ np.linalg.inv(T_cv_to_gazebo))

                    ### the following is a fix for camera orientation being turned 180 degrees about the z-axis
                    ### please do note that there's probably a better way to do this, but this works for now
                    R_fix = transforms3d.euler.euler2mat(0, 0, np.pi)
                    T_fix = np.eye(4)
                    T_fix[:3, :3] = R_fix

                    T_wc = T_wc @ T_fix 

                    # Extract position and quaternion
                    camera_pos = T_wc[:3, 3]
                    camera_rot = T_wc[:3, :3]
                    camera_quat = transforms3d.quaternions.mat2quat(camera_rot)

                    axes_length = 0.2

                    # Publish pose
                    pose_msg = PoseStamped()
                    pose_msg.header.stamp = self.get_clock().now().to_msg()
                    pose_msg.header.frame_id = 'world'
                    pose_msg.pose.position.x = float(camera_pos[0])
                    pose_msg.pose.position.y = float(camera_pos[1])
                    pose_msg.pose.position.z = float(camera_pos[2])
                    pose_msg.pose.orientation.x = float(camera_quat[1])
                    pose_msg.pose.orientation.y = float(camera_quat[2])
                    pose_msg.pose.orientation.z = float(camera_quat[3])
                    pose_msg.pose.orientation.w = float(camera_quat[0])
                    self.pose_pub.publish(pose_msg)

                    # Broadcast robot transform
                    t_robot = TransformStamped()
                    t_robot.header.stamp = self.get_clock().now().to_msg()
                    t_robot.header.frame_id = 'map'
                    t_robot.child_frame_id = 'base_link'
                    t_robot.transform.translation.x = float(camera_pos[0])
                    t_robot.transform.translation.y = float(camera_pos[1])
                    t_robot.transform.translation.z = 0.0
                    roll, pitch, yaw_robot = tf_transformations.euler_from_matrix(camera_rot)
                    quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw_robot)
                    t_robot.transform.rotation.x = quat[0]
                    t_robot.transform.rotation.y = quat[1]
                    t_robot.transform.rotation.z = quat[2]
                    t_robot.transform.rotation.w = quat[3]
                    self.br.sendTransform(t_robot)

                    # print(f"Robot pose from marker: x={cam_pos[0]:.2f}, y={cam_pos[1]:.2f}, yaw={yaw_robot:.2f}")


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

# import rclpy
# from rclpy.node import Node
# from geometry_msgs.msg import PoseStamped
# from sensor_msgs.msg import CompressedImage
# import cv2
# import numpy as np
# import tf_transformations
# from tf2_ros import TransformBroadcaster
# from geometry_msgs.msg import TransformStamped
# import yaml


# # def fix_yaw(yaw):
# #         # Flip yaw sign and normalize between [-pi, pi]
# #         yaw = -yaw
# #         while yaw > np.pi:
# #             yaw -= 2 * np.pi
# #         while yaw < -np.pi:
# #             yaw += 2 * np.pi
# #         return yaw

# class MarkerDetector(Node):
#     def __init__(self):
#         super().__init__('marker_detector')
#         self.subscription = self.create_subscription(
#             CompressedImage,
#             '/image_raw/compressed',
#             self.listener_callback,
#             10
#         )
#         self.pose_pub = self.create_publisher(PoseStamped, '/marker_pose', 10)
#         self.goal_pub = self.create_publisher(PoseStamped, '/goal_pose', 10)
#         self.br = TransformBroadcaster(self)

#         # Camera parameters
#         self.camera_matrix = np.array([
#             [920.7894, 0, 589.1690],
#             [0, 919.1768, 541.4353],
#             [0, 0, 1.0000]
#         ], dtype=np.float32)
#         self.dist_coeffs = np.array([[0.205245304646941], [-0.332544106955044], [0], [0], [0]], dtype=np.float32)

#         self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
#         self.parameters = cv2.aruco.DetectorParameters_create()

#         # Load marker positions and goals
#         yaml_path = '/home/monisha/turtlebot3_ws/src/my_aruco_detector/my_aruco_detector/aruco_marker_map.yaml'
#         with open(yaml_path, 'r') as f:
#             marker_map = yaml.safe_load(f)

#         self.marker_world_positions = {}
#         self.goal_markers = {}
#         for marker_id, data in marker_map.items():
#             if 'position' in data and 'yaw' in data:
#                 self.marker_world_positions[int(marker_id)] = [
#                     data['position'][0],
#                     data['position'][1],
#                     data['yaw']
#                 ]
#             if 'goal' in data:
#                 self.goal_markers[int(marker_id)] = data['goal']

#         self.marker_length = 0.03  # marker side length in meters

#     def listener_callback(self, msg):
#         try:
#             # Decode the compressed image
#             np_arr = np.frombuffer(msg.data, np.uint8)
#             frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)

#             if ids is not None:
#                 rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_length, self.camera_matrix, self.dist_coeffs)

#                 for i in range(len(ids)):
#                     marker_id = int(ids[i][0])

#                     # Set axis color and label color
#                     is_goal = marker_id in self.goal_markers
#                     axis_length = 0.02  # 2 cm
#                     color = (0, 255, 0) if is_goal else (255, 0, 0)  # Green for goal, Blue for regular

#                     # Draw the axis (OpenCV uses internal color logic, draw with default for all)
#                     cv2.aruco.drawAxis(frame, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], axis_length)

#                     # Draw the marker ID
#                     c = corners[i][0].mean(axis=0).astype(int)
#                     label = f"Goal ID: {marker_id}" if is_goal else f"ID: {marker_id}"
#                     cv2.putText(frame, label, (c[0], c[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


#                     # First, if it is a goal marker
#                     if marker_id in self.goal_markers:
#                         goal_x, goal_y = self.goal_markers[marker_id]
#                         goal_msg = PoseStamped()
#                         goal_msg.header.stamp = self.get_clock().now().to_msg()
#                         goal_msg.header.frame_id = 'map'
#                         goal_msg.pose.position.x = goal_x
#                         goal_msg.pose.position.y = goal_y
#                         goal_msg.pose.position.z = 0.0
#                         goal_msg.pose.orientation.w = 1.0  # no orientation for goal
#                         self.goal_pub.publish(goal_msg)
#                         continue  # DO NOT reset pose with goal markers

#                     if marker_id not in self.marker_world_positions:
#                         continue

#                     # For pillar markers: do localization reset
#                     tvec = tvecs[i].reshape((3, 1))
#                     rvec = rvecs[i].reshape((3, 1))

#                     # marker in camera frame
#                     R_m_c, _ = cv2.Rodrigues(rvec)
#                     T_m_c = np.eye(4)
#                     T_m_c[:3, :3] = R_m_c
#                     T_m_c[:3, 3] = tvec[:, 0]

#                     # Invert to get camera in marker frame
#                     T_c_m = np.linalg.inv(T_m_c)

#                     # Define correct camera to robot transform
#                     T_r_c = np.array([
#                         [ 0,  0, 1, 0],   # x_r = z_c
#                         [ -1,  0, 0, 0],   # y_r = x_c
#                         [ 0,  -1, 0, 0],   # z_r = y_c
#                         [ 0,  0, 0, 1]
#                     ])

#                     # robot in marker frame
#                     T_r_m = T_r_c @ T_c_m

#                     # Get marker in world from YAML
#                     x, y, yaw = self.marker_world_positions[marker_id]
#                     T_w_m = np.eye(4)
#                     T_w_m[:3, :3] = tf_transformations.euler_matrix(0, 0, yaw)[:3, :3]
#                     T_w_m[:3, 3] = [x, y, 0.0]

#                     # Final: robot in world
#                     T_w_r = T_w_m @ T_r_m

#                     # Extract position and yaw
#                     cam_pos = T_w_r[:3, 3]
#                     roll, pitch, yaw_robot = tf_transformations.euler_from_matrix(T_w_r)

#                     # roll, pitch, yaw_robot = tf_transformations.euler_from_matrix(T_w_r)
#                     # yaw_robot = fix_yaw(yaw_robot)

#                     pose_msg = PoseStamped()
#                     pose_msg.header.stamp = self.get_clock().now().to_msg()
#                     pose_msg.header.frame_id = 'map'
#                     pose_msg.pose.position.x = float(cam_pos[0])
#                     pose_msg.pose.position.y = float(cam_pos[1])
#                     pose_msg.pose.position.z = 0.0
#                     pose_msg.pose.orientation.z = np.sin(yaw_robot / 2)
#                     pose_msg.pose.orientation.w = np.cos(yaw_robot / 2)
#                     self.pose_pub.publish(pose_msg)

#                     # Broadcast robot transform
#                     t_robot = TransformStamped()
#                     t_robot.header.stamp = self.get_clock().now().to_msg()
#                     t_robot.header.frame_id = 'map'
#                     t_robot.child_frame_id = 'base_link'
#                     t_robot.transform.translation.x = float(cam_pos[0])
#                     t_robot.transform.translation.y = float(cam_pos[1])
#                     t_robot.transform.translation.z = 0.0
#                     quat = tf_transformations.quaternion_from_euler(roll, pitch, yaw_robot)
#                     t_robot.transform.rotation.x = quat[0]
#                     t_robot.transform.rotation.y = quat[1]
#                     t_robot.transform.rotation.z = quat[2]
#                     t_robot.transform.rotation.w = quat[3]
#                     self.br.sendTransform(t_robot)

#             cv2.imshow("ArUco Marker Detection", frame)
#             cv2.waitKey(1)

#         except Exception as e:
#             self.get_logger().error(f"Error processing image: {e}")

# def main(args=None):
#     rclpy.init(args=args)
#     marker_detector = MarkerDetector()
#     rclpy.spin(marker_detector)
#     marker_detector.destroy_node()
#     rclpy.shutdown()
#     cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main()

