import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
import numpy as np
import math
import random

class ParticleFilterNode(Node):
    def __init__(self):
        super().__init__('particle_filter_node')

        self.num_particles = 100
        self.particles = np.zeros((self.num_particles, 5))  # [x, y, theta, v, omega]
        self.weights = np.ones(self.num_particles) / self.num_particles

        self.last_time = self.get_clock().now()

        self.imu_data = None

        self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.create_subscription(PoseStamped, '/marker_pose', self.marker_callback, 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/fused_pose', 10)

        self.timer = self.create_timer(0.05, self.predict)

    def normalize_angle(self, angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi

    def imu_callback(self, msg):
        self.imu_data = msg

    def predict(self):
        now = self.get_clock().now()
        dt = (now - self.last_time).nanoseconds * 1e-9
        self.last_time = now
        if dt <= 0 or dt > 1.0:
            return

        if self.imu_data is None:
            return

        ax = self.imu_data.linear_acceleration.x
        omega_meas = self.imu_data.angular_velocity.z

        for i in range(self.num_particles):
            x, y, theta, v, omega = self.particles[i]

            v += ax * dt + np.random.normal(0, 0.1)
            omega = omega_meas + np.random.normal(0, 0.05)
            theta += omega * dt
            theta = self.normalize_angle(theta)

            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt

            self.particles[i] = [x, y, theta, v, omega]

        self.publish_pose()

    def odom_callback(self, msg):
        z = np.array([msg.pose.pose.position.x,
                      msg.pose.pose.position.y,
                      msg.twist.twist.linear.x,
                      msg.twist.twist.angular.z])
        R = np.diag([0.05, 0.05, 0.1, 0.05])
        self.update(z, R, indices=[0, 1, 3, 4])  # x, y, v, omega

    def marker_callback(self, msg):
        x = msg.pose.position.x
        y = msg.pose.position.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w
        theta = 2 * np.arctan2(qz, qw)

        z = np.array([x, y, theta])
        R = np.diag([0.02, 0.02, 0.05])
        self.update(z, R, indices=[0, 1, 2])  # x, y, theta

    def update(self, z, R, indices):
        for i in range(self.num_particles):
            diff = z - self.particles[i, indices]
            if 2 in indices:
                # Normalize angle difference
                idx = indices.index(2)
                diff[idx] = self.normalize_angle(diff[idx])

            likelihood = np.exp(-0.5 * diff.T @ np.linalg.inv(R) @ diff)
            self.weights[i] *= likelihood + 1e-300  # avoid 0 weight

        self.weights /= np.sum(self.weights)
        self.resample()

    def resample(self):
        new_particles = []
        cumsum = np.cumsum(self.weights)
        step = 1.0 / self.num_particles
        r = random.uniform(0, step)
        i = 0

        for _ in range(self.num_particles):
            while r > cumsum[i]:
                i += 1
            new_particles.append(self.particles[i].copy())

            r += step

        self.particles = np.array(new_particles)
        self.weights = np.ones(self.num_particles) / self.num_particles

    def publish_pose(self):
        best_index = np.argmax(self.weights)
        best = self.particles[best_index]

        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.position.x = float(best[0])
        pose_msg.pose.position.y = float(best[1])
        pose_msg.pose.position.z = 0.0
        pose_msg.pose.orientation.z = math.sin(best[2] / 2.0)
        pose_msg.pose.orientation.w = math.cos(best[2] / 2.0)

        self.pose_pub.publish(pose_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ParticleFilterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
