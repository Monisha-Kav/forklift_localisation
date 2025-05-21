# localization.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_aruco_detector',
            executable='marker_pose_localise',
            name='aruco_detector',
            output='screen'
        ),
        Node(
            package='my_aruco_detector',
            executable='kf_full_sensor_fusion_m',
            name='sensor_fusion',
            output='screen'
        ),
        Node(
            package='my_aruco_detector',
            executable='new_forklift_display',
            name='forklift_display',
            output='screen'
        ),

    ])
