#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class CmdVelSubscriber(Node):
    def __init__(self):
        super().__init__('cmd_vel_subscriber')
        # /cmd_vel の Subscriber を作成
        self.sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        # 起動時のログも出さない
        # self.get_logger().info("CmdVelSubscriber started")

    def cmd_vel_callback(self, msg: Twist):
        # ログ出力しない
        pass

def main():
    rclpy.init()
    node = CmdVelSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
