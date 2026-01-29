#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from nav_msgs.msg import Odometry
import tf2_ros

def yaw_to_quat(yaw: float):
    # planar yaw -> quaternion
    half = 0.5 * yaw
    return (0.0, 0.0, math.sin(half), math.cos(half))

class CmdVelOdom(Node):
    """
    Very simple dead-reckoning odometry from /cmd_vel.
    Publishes:
      - nav_msgs/Odometry on /odom
      - TF odom -> base_link
    NOTE: drifts over time (no sensor correction), but enough to create 'odom' frame for ego compensation.
    """
    def __init__(self):
        super().__init__('cmdvel_odom')

        # params
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('odom_topic', '/odom')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('rate_hz', 50.0)
        self.declare_parameter('cmd_timeout_sec', 0.3)

        self.cmd_topic = str(self.get_parameter('cmd_topic').value)
        self.odom_topic = str(self.get_parameter('odom_topic').value)
        self.odom_frame = str(self.get_parameter('odom_frame').value)
        self.base_frame = str(self.get_parameter('base_frame').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.cmd_timeout = float(self.get_parameter('cmd_timeout_sec').value)

        # state
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.v = 0.0
        self.w = 0.0
        self.last_cmd_time = self.get_clock().now()

        self.last_tick_time = self.get_clock().now()

        # ros I/O
        self.sub = self.create_subscription(Twist, self.cmd_topic, self.cb_cmd, 10)
        self.pub_odom = self.create_publisher(Odometry, self.odom_topic, 10)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        period = 1.0 / max(self.rate_hz, 1.0)
        self.timer = self.create_timer(period, self.tick)

        self.get_logger().info(
            f"CmdVelOdom started. cmd={self.cmd_topic} odom={self.odom_topic} "
            f"tf: {self.odom_frame}->{self.base_frame}"
        )

    def cb_cmd(self, msg: Twist):
        self.v = float(msg.linear.x)
        self.w = float(msg.angular.z)
        self.last_cmd_time = self.get_clock().now()

    def tick(self):
        now = self.get_clock().now()
        dt = (now - self.last_tick_time).nanoseconds * 1e-9
        if dt <= 0.0:
            return
        self.last_tick_time = now

        # if cmd is stale, assume stop
        cmd_age = (now - self.last_cmd_time).nanoseconds * 1e-9
        v = self.v if cmd_age <= self.cmd_timeout else 0.0
        w = self.w if cmd_age <= self.cmd_timeout else 0.0

        # integrate (planar)
        self.yaw += w * dt
        self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))  # wrap

        self.x += v * math.cos(self.yaw) * dt
        self.y += v * math.sin(self.yaw) * dt

        # publish odom msg
        odom = Odometry()
        odom.header.stamp = now.to_msg()
        odom.header.frame_id = self.odom_frame
        odom.child_frame_id = self.base_frame
        odom.pose.pose.position.x = self.x
        odom.pose.pose.position.y = self.y
        odom.pose.pose.position.z = 0.0
        qx, qy, qz, qw = yaw_to_quat(self.yaw)
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x = v
        odom.twist.twist.angular.z = w
        self.pub_odom.publish(odom)

        # publish TF odom -> base_link
        tfm = TransformStamped()
        tfm.header.stamp = odom.header.stamp
        tfm.header.frame_id = self.odom_frame
        tfm.child_frame_id = self.base_frame
        tfm.transform.translation.x = self.x
        tfm.transform.translation.y = self.y
        tfm.transform.translation.z = 0.0
        tfm.transform.rotation.x = qx
        tfm.transform.rotation.y = qy
        tfm.transform.rotation.z = qz
        tfm.transform.rotation.w = qw
        self.tf_broadcaster.sendTransform(tfm)

def main(args=None):
    rclpy.init(args=args)
    node = CmdVelOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
