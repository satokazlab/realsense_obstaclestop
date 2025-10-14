#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

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
        self.last_log_time = 0.0  # ログ間隔制御用
        # 起動ログは1回だけ INFO
        self.get_logger().info("CmdVelSubscriber started (logs throttled)")

    def cmd_vel_callback(self, msg: Twist):
        """
        /cmd_vel を受信するコールバック。
        ログは 1 秒に 1 回だけ出力して大量表示を防ぐ。
        レベルを DEBUG にすることで launch 実行時に出ないようにする。
        """
        now = time.time()
        if now - self.last_log_time > 1.0:
            self.get_logger().debug(
                f"Received cmd_vel -> linear.x: {msg.linear.x:.2f}, angular.z: {msg.angular.z:.2f}"
            )
            self.last_log_time = now

def main():
    rclpy.init()
    node = CmdVelSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
