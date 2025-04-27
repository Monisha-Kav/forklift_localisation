# imu_calibration_node.py

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

class IMUCalibrator(Node):
    def __init__(self):
        super().__init__('imu_calibrator')
        self.subscriber = self.create_subscription(Imu, '/imu', self.imu_callback, 10)
        self.publisher = self.create_publisher(Imu, '/imu/calibrated', 10)

        # Measured biases
        self.gyro_bias = {'x': 0.02, 'y': -0.01, 'z': 0.00}
        self.accel_bias = {'x': 0.2, 'y': -0.03, 'z': 10}  # ignore gravity

    def imu_callback(self, msg):
        calibrated_msg = Imu()
        calibrated_msg.header = msg.header

        calibrated_msg.angular_velocity.x = msg.angular_velocity.x - self.gyro_bias['x']
        calibrated_msg.angular_velocity.y = msg.angular_velocity.y - self.gyro_bias['y']
        calibrated_msg.angular_velocity.z = msg.angular_velocity.z - self.gyro_bias['z']

        calibrated_msg.linear_acceleration.x = msg.linear_acceleration.x - self.accel_bias['x']
        calibrated_msg.linear_acceleration.y = msg.linear_acceleration.y - self.accel_bias['y']
        calibrated_msg.linear_acceleration.z = msg.linear_acceleration.z - self.accel_bias['z']

        self.publisher.publish(calibrated_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IMUCalibrator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

