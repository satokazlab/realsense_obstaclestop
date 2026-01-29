#!/usr/bin/env python3
"""
rgb_object_locator.py

- Subscribe: (none; reads RealSense directly)
- Publish: /rgb_detections_markers (visualization_msgs/MarkerArray)
Each marker.pose.position will contain the deprojected 3D point in the camera depth frame
(so tracker that expects markers in the same frame can directly match by euclidean distance).
"""

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import MarkerArray, Marker
import numpy as np
import pyrealsense2 as rs
import cv2
import time

class RGBObjectLocator(Node):
    def __init__(self):
        super().__init__('rgb_object_locator')
        # params
        self.declare_parameter('publish_topic', '/rgb_detections_markers')
        self.declare_parameter('camera_frame', 'camera_depth_optical_frame')
        self.declare_parameter('min_area', 500)          # contour area threshold
        self.declare_parameter('hsv_lower', [0, 120, 70])   # default red low
        self.declare_parameter('hsv_upper', [10, 255, 255]) # default red high (use two ranges for red)
        self.declare_parameter('hsv_lower2', [170, 120, 70])
        self.declare_parameter('hsv_upper2', [180, 255, 255])
        self.declare_parameter('publish_rate', 10)       # Hz
        self.declare_parameter('marker_scale_z', 0.05)

        self.pub_topic = str(self.get_parameter('publish_topic').value)
        self.camera_frame = str(self.get_parameter('camera_frame').value)
        self.min_area = int(self.get_parameter('min_area').value)
        self.hsv_lower = np.array(self.get_parameter('hsv_lower').value, dtype=np.uint8)
        self.hsv_upper = np.array(self.get_parameter('hsv_upper').value, dtype=np.uint8)
        self.hsv_lower2 = np.array(self.get_parameter('hsv_lower2').value, dtype=np.uint8)
        self.hsv_upper2 = np.array(self.get_parameter('hsv_upper2').value, dtype=np.uint8)
        self.pub_rate = float(self.get_parameter('publish_rate').value)
        self.marker_scale_z = float(self.get_parameter('marker_scale_z').value)

        self.pub = self.create_publisher(MarkerArray, self.pub_topic, 10)

        # RealSense init
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(cfg)
        # get depth scale
        self.depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = self.depth_sensor.get_depth_scale()
        # align depth to color
        self.align = rs.align(rs.stream.color)
        # intrinsics for deprojection (will fetch after first frames)
        self.intrinsics = None

        self.get_logger().info(f"rgb_object_locator started, publishing {self.pub_topic}")

        # run loop in timer
        self.timer = self.create_timer(1.0 / self.pub_rate, self.timer_cb)
        self.marker_id_seq = 0

    def timer_cb(self):
        frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        if frames is None:
            return
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return
        if self.intrinsics is None:
            color_stream = self.profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # convert to HSV
        hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        mask2 = cv2.inRange(hsv, self.hsv_lower2, self.hsv_upper2)
        mask = cv2.bitwise_or(mask1, mask2)

        # morphology
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        markers = MarkerArray()
        now = self.get_clock().now().to_msg()

        mid = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            cx = int(x + w/2)
            cy = int(y + h/2)
            # get depth in meters (handle zero-depth by local search)
            z = depth_image[cy, cx] * self.depth_scale
            if z == 0.0:
                # search nearby for a nonzero depth
                sz = 5
                patch = depth_image[max(0,cy-sz):min(depth_image.shape[0],cy+sz+1),
                                    max(0,cx-sz):min(depth_image.shape[1],cx+sz+1)]
                nz = patch[patch>0]
                if nz.size > 0:
                    z = float(np.median(nz) * self.depth_scale)
                else:
                    continue  # skip if no depth

            # deproject to 3D point (meters) in camera depth frame
            px = [cx, cy]
            point = rs.rs2_deproject_pixel_to_point(self.intrinsics, px, z)
            # create marker
            m = Marker()
            m.header.stamp = now
            m.header.frame_id = self.camera_frame
            m.ns = 'rgb_detections'
            m.id = mid
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.pose.position.x = float(point[0])
            m.pose.position.y = float(point[1])
            m.pose.position.z = float(point[2])
            m.pose.orientation.w = 1.0
            m.scale.x = float(w) / 1000.0  # just small marker scaling heuristics
            m.scale.y = float(h) / 1000.0
            m.scale.z = float(self.marker_scale_z)
            # show red color for detected object
            m.color.r, m.color.g, m.color.b, m.color.a = (1.0, 0.2, 0.2, 0.9)
            markers.markers.append(m)
            mid += 1

        if len(markers.markers) > 0:
            self.pub.publish(markers)

    def destroy(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RGBObjectLocator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
