import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
import smbus
import time

class ADXL345Node(Node):
    def __init__(self):
        super().__init__('adxl345_node')
        self.publisher = self.create_publisher(Imu, '/adxl_imu', 10)
        self.bus = smbus.SMBus(1)
        self.address = 0x53

        # Initialize ADXL345
        self.bus.write_byte_data(self.address, 0x2D, 0x08)  # Power on
        self.bus.write_byte_data(self.address, 0x31, 0x08)  # Set full resolution and +-2g

        self.timer = self.create_timer(0.05, self.read_acceleration)  # 20 Hz

    def read_acceleration(self):
        def read_word(adr):
            low = self.bus.read_byte_data(self.address, adr)
            high = self.bus.read_byte_data(self.address, adr + 1)
            val = (high << 8) + low
            if val > 32767:
                val -= 65536
            return val

        imu_msg = Imu()
        imu_msg.header.stamp = self.get_clock().now().to_msg()
        imu_msg.header.frame_id = 'base_link'

        scale_factor = 0.0039  # g per LSB

        ax = read_word(0x32) * scale_factor * 9.81
        ay = read_word(0x34) * scale_factor * 9.81
        az = read_word(0x36) * scale_factor * 9.81

        imu_msg.linear_acceleration.x = ax
        imu_msg.linear_acceleration.y = ay
        imu_msg.linear_acceleration.z = az

        # Leave angular_velocity empty (no gyro in ADXL345)

        self.publisher.publish(imu_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ADXL345Node()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

