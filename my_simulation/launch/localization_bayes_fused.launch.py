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
            executable='full_sensor_fusion_fixed',
            name='full_sensor_fusion_fixed',
            output='screen'
        ),
        Node(
            package='my_simulation',
            executable='new_forklift_display_sim',
            name='new_forklift_display_sim',
            output='screen'
        ),
    ])

