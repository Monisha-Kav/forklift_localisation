from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare

import os

def generate_launch_description():
    world_path = '/home/monisha/simulation_ws/src/my_simulation/worlds/test_world.world'

    return LaunchDescription([
        ExecuteProcess(
            cmd=[
                'gazebo',
                '--verbose',
                world_path,
                '-s', 'libgazebo_ros_init.so',
                '-s', 'libgazebo_ros_factory.so',
            ],
            output='screen'
        ),
    ])

