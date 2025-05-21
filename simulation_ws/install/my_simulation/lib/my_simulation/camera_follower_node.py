#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from gazebo_msgs.srv import SetEntityState
from gazebo_msgs.msg import EntityState
from geometry_msgs.msg import Pose, Quaternion
from tf_transformations import euler_from_quaternion
import math

class CameraFollower(Node):
    def __init__(self):
        super().__init__('camera_follower')
        self.sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.cli = self.create_client(SetEntityState, '/set_entity_state')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for /set_entity_state service...')
        self.initialized = False

    def odom_callback(self, msg):
        if not self.initialized:
            self.get_logger().info('Received initial pose, starting camera tracking...')
            self.initialized = True

        if self.initialized:
            state = EntityState()
            state.name = 'camera'  # Gazebo camera model name (update if different)
            
            # Get robot position from /odom message
            robot_position = msg.pose.pose.position
            robot_orientation = msg.pose.pose.orientation
            
            # Extract yaw from robot's orientation (quaternion to Euler)
            _, _, yaw = euler_from_quaternion(
                [robot_orientation.x, robot_orientation.y, robot_orientation.z, robot_orientation.w]
            )
            
            # create camera pose
            state.pose = Pose()
            state.pose.position.x = msg.pose.pose.position.x + 0.045
            state.pose.position.y = msg.pose.pose.position.y   # Offset towards the back 
            state.pose.position.z = msg.pose.pose.position.z + 0.105 # Elevated

            # set camera orientation to match robot's yaw
            quaternion = self.yaw_to_quaternion(yaw)
            state.pose.orientation = quaternion

            req = SetEntityState.Request()
            req.state = state
            self.cli.call_async(req)
            
    def yaw_to_quaternion(self, yaw):
        """Convert yaw to quaternion."""
        q = Quaternion()
        q.x = 0.0
        q.y = 0.0
        q.z = math.sin(yaw / 2.0)
        q.w = math.cos(yaw / 2.0)
        return q

def main(args=None):
    rclpy.init(args=args)
    node = CameraFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

