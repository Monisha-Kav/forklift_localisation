# localization.launch.py
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='my_simulation',
            executable='aruco_live_detector',
            name='aruco_live_detector',
            output='screen'
        ),
        Node(
            package='my_simulation',
            executable='sensor_fusion_node_sim',
            name='sensor_fusion_node_sim',
            output='screen'
        ),
        Node(
            package='my_simulation',
            executable='forklift_display_sim',
            name='forklift_display_sim',
            output='screen'
        ),
    ])

