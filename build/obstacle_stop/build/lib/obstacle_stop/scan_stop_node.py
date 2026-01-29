#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data

class ScanStopNode(Node):
    def __init__(self):
        super().__init__('scan_stop_node')

        # --- パラメータ ---
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('min_distance', 0.3)    # この距離以内で停止
        self.declare_parameter('forward_speed', 0.2)   # 障害物なし時の前進速度
        self.declare_parameter('publish_rate', 10.0)   # Hz

        self.scan_topic = self.get_parameter('scan_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.min_distance = float(self.get_parameter('min_distance').value)
        self.forward_speed = float(self.get_parameter('forward_speed').value)
        self.publish_rate = float(self.get_parameter('publish_rate').value)

        # --- Publisher / Subscriber ---
        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.sub_scan = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data
        )

        self.get_logger().info(f"Subscribe: {self.scan_topic} | Publish: {self.cmd_vel_topic}")

        # 最新のTwistを保持して定期publish
        self.latest_twist = Twist()
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.publish_cmd)

    def scan_callback(self, msg: LaserScan):
        if not msg.ranges:
            self.get_logger().warn("No scan data -> STOP")
            self.latest_twist.linear.x = 0.0
            return

        # 有効距離のみ抽出
        valid_ranges = [r for r in msg.ranges if msg.range_min < r < msg.range_max]
        if not valid_ranges:
            self.get_logger().warn("No valid scan data -> STOP")
            self.latest_twist.linear.x = 0.0
            return

        min_range = min(valid_ranges)
        twist = Twist()
        if min_range < self.min_distance:
            twist.linear.x = 0.0
            self.get_logger().info(f"Obstacle detected at {min_range:.2f} m -> STOP")
        else:
            twist.linear.x = self.forward_speed
            self.get_logger().info(f"Path clear, min distance {min_range:.2f} m -> GO")

        self.latest_twist = twist

    def publish_cmd(self):
        # 定期的に最新のTwistをpublish
        self.pub_cmd.publish(self.latest_twist)

def main():
    rclpy.init()
    node = ScanStopNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
