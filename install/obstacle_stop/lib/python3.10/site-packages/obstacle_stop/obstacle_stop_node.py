# obstacle_stop/obstacle_stop_node.py
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from geometry_msgs.msg import Twist
from rclpy.parameter import Parameter

class ObstacleStopNode(Node):
    def __init__(self):
        super().__init__('obstacle_stop_node')

        # -------- パラメータ（必要なら起動時に上書き可） --------
        self.declare_parameter('pointcloud_topic', '/camera/camera/depth/color/points')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('min_distance', 0.3)      # 前方この距離以内で障害物と判定[m]
        self.declare_parameter('x_half', 0.3)            # 左右方向の関心範囲 ±x_half [m]
        self.declare_parameter('y_half', 0.3)            # 上下方向の関心範囲 ±y_half [m]
        self.declare_parameter('forward_speed', 0.2)     # 障害物なし時の前進速度[m/s]
        self.declare_parameter('sample_step', 10)        # 点群の間引き（計算軽量化）
        self.declare_parameter('stop_hold_time', 1.0)    # 検知後、停止を維持する秒数

        self.pointcloud_topic = self.get_parameter('pointcloud_topic').value
        self.cmd_vel_topic     = self.get_parameter('cmd_vel_topic').value
        self.min_distance      = float(self.get_parameter('min_distance').value)
        self.x_half            = float(self.get_parameter('x_half').value)
        self.y_half            = float(self.get_parameter('y_half').value)
        self.forward_speed     = float(self.get_parameter('forward_speed').value)
        self.sample_step       = int(self.get_parameter('sample_step').value)
        self.stop_hold_time    = float(self.get_parameter('stop_hold_time').value)

        # Publisher / Subscriber
        self.pub_cmd = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.sub_pc  = self.create_subscription(
            PointCloud2, self.pointcloud_topic, self.pc_callback, qos_profile_sensor_data
        )

        self._last_stop_time = None
        self.get_logger().info(
            f"Subscribe: {self.pointcloud_topic} | Publish: {self.cmd_vel_topic}"
        )

    def pc_callback(self, msg: PointCloud2):
        obstacle = False

        # RealSenseの PointCloud2 は通常「光学フレーム」なので
        # z: 前方(奥)/ x: 右(左右)/ y: 下(上下) が正方向になることが多い点に注意
        for i, (x, y, z) in enumerate(pc2.read_points(msg, field_names=("x","y","z"), skip_nans=True)):
            if self.sample_step > 1 and (i % self.sample_step != 0):
                continue
            # 関心領域: 中央付近かつ前方 min_distance 以内
            if (abs(x) < self.x_half) and (abs(y) < self.y_half) and (0.0 < z < self.min_distance):
                obstacle = True
                break

        now = self.get_clock().now()

        if obstacle:
            self._last_stop_time = now
            self.publish_speed(0.0)
            self.get_logger().debug("Obstacle detected -> STOP")
        else:
            # 直前に検知していたら少し停止を保持してから再加速
            if self._last_stop_time is not None:
                elapsed = (now - self._last_stop_time).nanoseconds / 1e9
                if elapsed < self.stop_hold_time:
                    self.publish_speed(0.0)
                    return
            self.publish_speed(self.forward_speed)

    def publish_speed(self, v):
        twist = Twist()
        twist.linear.x = float(v)
        self.pub_cmd.publish(twist)

def main():
    rclpy.init()
    node = ObstacleStopNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
