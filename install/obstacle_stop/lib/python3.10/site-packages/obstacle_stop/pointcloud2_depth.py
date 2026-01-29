#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, LaserScan
import sensor_msgs_py.point_cloud2 as pc2
import math

class PCL2ToLaserScan(Node):
    def __init__(self):
        super().__init__('pcl2_to_laserscan')

        # params (変更可)
        self.input_topic = self.declare_parameter('input_topic', '/camera/camera/depth/color/points').value
        self.output_topic = self.declare_parameter('output_topic', '/scan').value
        self.angle_min = self.declare_parameter('angle_min', -math.pi/2).value
        self.angle_max = self.declare_parameter('angle_max', math.pi/2).value
        self.angle_increment = self.declare_parameter('angle_increment', math.radians(0.5)).value
        self.range_min = self.declare_parameter('range_min', 0.1).value
        self.range_max = self.declare_parameter('range_max', 10.0).value

        # 'euclidean' | 'forward'  ('euclidean' = sqrt(x^2 + lateral^2), 'forward' = depth along forward_axis)
        self.depth_mode = self.declare_parameter('depth_mode', 'euclidean').value
        # which axis is "forward" in your pointcloud: 'x' (REP103 style) or 'z' (camera optical)
        self.forward_axis = self.declare_parameter('forward_axis', 'z').value

        self.sub = self.create_subscription(PointCloud2, self.input_topic, self.callback, 10)
        self.pub = self.create_publisher(LaserScan, self.output_topic, 10)

        # precompute bin count
        self.num_bins = max(1, int(math.floor((self.angle_max - self.angle_min) / self.angle_increment)) + 1)
        self.get_logger().info(f"PCL2->LaserScan: input={self.input_topic}, output={self.output_topic}, bins={self.num_bins}")

    def callback(self, msg: PointCloud2):
        # initialize ranges with +inf
        ranges = [float('inf')] * self.num_bins

        # iterate points (x,y,z)
        for p in pc2.read_points(msg, field_names=('x','y','z'), skip_nans=True):
            x, y, z = p

            # choose angle and depth depending on forward axis
            if self.forward_axis == 'x':
                # forward = x, lateral = y
                angle = math.atan2(y, x)  # -pi .. +pi
                if self.depth_mode == 'forward':
                    depth = x if x >= 0.0 else None
                else:
                    depth = math.hypot(x, y)
            else:
                # assume forward = z (camera optical): lateral = x (left/right)
                # angle measured around forward axis: use atan2(x, z)
                angle = math.atan2(x, z)
                if self.depth_mode == 'forward':
                    depth = z if z >= 0.0 else None
                else:
                    depth = math.hypot(z, x)

            if depth is None:
                continue
            if not (self.range_min <= depth <= self.range_max):
                continue

            # bin index
            if angle < self.angle_min or angle > self.angle_max:
                continue
            index = int((angle - self.angle_min) / self.angle_increment)
            if index < 0 or index >= self.num_bins:
                continue

            # keep minimum (closest) reading per angle
            if depth < ranges[index]:
                ranges[index] = depth

        # Build LaserScan
        scan = LaserScan()
        scan.header = msg.header  # stamp/frame from pointcloud (you can override if needed)
        scan.angle_min = self.angle_min
        scan.angle_max = self.angle_max
        scan.angle_increment = self.angle_increment
        scan.range_min = self.range_min
        scan.range_max = self.range_max
        # keep inf for no-data bins (rviz treats inf as no reading)
        scan.ranges = [r if r != float('inf') else float('inf') for r in ranges]

        self.pub.publish(scan)

def main(args=None):
    rclpy.init(args=args)
    node = PCL2ToLaserScan()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
