# localization.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
         Node(
            package='my_aruco_detector',
            executable='imu_calibration_node',
            name='imu_calibration_node',
            output='screen'
        ),
        Node(
            package='my_aruco_detector',
            executable='marker_pose_localise',
            name='aruco_detector',
            output='screen'
        ),
        Node(
            package='my_aruco_detector',
            executable='sensor_fusion_node_fixed',
            name='sensor_fusion_fixed',
            output='screen'
        ),
        Node(
            package='my_aruco_detector',
            executable='forklift_display',
            name='forklift_display',
            output='screen'
        ),
    ])
