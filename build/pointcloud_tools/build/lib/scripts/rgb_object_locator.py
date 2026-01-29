#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import Header
from cv_bridge import CvBridge
import numpy as np
import cv2
import time


def stamp_is_valid(stamp) -> bool:
    try:
        return (stamp.sec != 0) or (stamp.nanosec != 0)
    except Exception:
        return False


def create_pc2_xyz(points_xyz: np.ndarray, frame_id: str, stamp=None):
    header = Header()
    header.frame_id = frame_id
    if stamp is None or not stamp_is_valid(stamp):
        header.stamp = rclpy.time.Time(seconds=int(time.time())).to_msg()
    else:
        header.stamp = stamp
    return pc2.create_cloud_xyz32(header, points_xyz.tolist())


class RGBMotionPointCloudFilter(Node):
    """
    Subscribe:
      - raw dynamic pointcloud (PointCloud2): /pointcloud_diff_comp_raw
      - RGB motion mask (Image mono8):        /rgb_motion_mask
      - camera intrinsics (CameraInfo):       /camera/color/camera_info (or aligned depth cam_info)

    Output:
      - filtered dynamic pointcloud:          /pointcloud_diff_comp

    Core idea:
      project each 3D point to image pixel (u,v) using pinhole model,
      keep the point only if motion_mask[v,u] != 0

    Notes:
      - Works best if pointcloud is already in the same camera optical frame as the camera model used
      - If your mask is built from /camera/color/image_raw, use color camera_info.
      - If you use aligned depth to color, still use color camera_info.
    """

    def __init__(self):
        super().__init__('rgb_motion_pointcloud_filter')

        # topics
        self.declare_parameter('pc_in_topic', '/pointcloud_diff_comp_raw')
        self.declare_parameter('mask_topic', '/rgb_motion_mask')
        self.declare_parameter('camera_info_topic', '/camera/color/camera_info')
        self.declare_parameter('pc_out_topic', '/pointcloud_diff_comp')

        # mask processing
        self.declare_parameter('mask_dilate_ksize', 5)   # 0 => no dilate
        self.declare_parameter('mask_hold_frames', 2)    # OR masks for N frames (stability)
        self.declare_parameter('mask_min_value', 1)      # keep if mask pixel >= this

        # projection / filtering
        self.declare_parameter('min_z', 0.15)            # ignore points too close
        self.declare_parameter('max_z', 5.0)             # ignore too far (safety)
        self.declare_parameter('max_points', 60000)      # cap for speed (deterministic by nearest-to-origin)

        # read params
        self.pc_in_topic = str(self.get_parameter('pc_in_topic').value)
        self.mask_topic = str(self.get_parameter('mask_topic').value)
        self.camera_info_topic = str(self.get_parameter('camera_info_topic').value)
        self.pc_out_topic = str(self.get_parameter('pc_out_topic').value)

        self.mask_dilate_ksize = int(self.get_parameter('mask_dilate_ksize').value)
        self.mask_hold_frames = int(self.get_parameter('mask_hold_frames').value)
        self.mask_min_value = int(self.get_parameter('mask_min_value').value)

        self.min_z = float(self.get_parameter('min_z').value)
        self.max_z = float(self.get_parameter('max_z').value)
        self.max_points = int(self.get_parameter('max_points').value)

        # intrinsics
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.bridge = CvBridge()

        # latest mask + hold buffer
        self.mask_hold = []   # list of mono8 np arrays
        self.latest_mask = None
        self.latest_mask_stamp = None

        # subs/pubs
        self.create_subscription(CameraInfo, self.camera_info_topic, self.cb_info, 10)
        self.create_subscription(Image, self.mask_topic, self.cb_mask, 10)
        self.create_subscription(PointCloud2, self.pc_in_topic, self.cb_pc, 10)

        self.pub = self.create_publisher(PointCloud2, self.pc_out_topic, 10)

        self.get_logger().info("RGBMotionPointCloudFilter started")
        self.get_logger().info(f"pc_in={self.pc_in_topic} mask={self.mask_topic} cam_info={self.camera_info_topic}")
        self.get_logger().info(f"pc_out={self.pc_out_topic} dilate={self.mask_dilate_ksize} hold={self.mask_hold_frames}")

    def cb_info(self, msg: CameraInfo):
        # K = [fx 0 cx; 0 fy cy; 0 0 1]
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def cb_mask(self, msg: Image):
        # Expect mono8 mask (0/255). If not, we try to convert.
        mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        if mask is None:
            return

        # ensure uint8 2D
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        if mask.dtype != np.uint8:
            mask = np.clip(mask, 0, 255).astype(np.uint8)

        # optional dilate to tolerate projection mismatch
        if self.mask_dilate_ksize and self.mask_dilate_ksize > 1:
            k = self.mask_dilate_ksize
            kernel = np.ones((k, k), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        # hold N frames (OR)
        self.mask_hold.append(mask)
        if len(self.mask_hold) > max(1, self.mask_hold_frames):
            self.mask_hold = self.mask_hold[-self.mask_hold_frames:]

        merged = self.mask_hold[0].copy()
        for m in self.mask_hold[1:]:
            if m.shape == merged.shape:
                merged = cv2.bitwise_or(merged, m)
            else:
                # shape mismatch: drop old buffer to avoid wrong gating
                self.mask_hold = [mask]
                merged = mask.copy()
                break

        self.latest_mask = merged
        self.latest_mask_stamp = msg.header.stamp

    def _pc2_to_xyz(self, msg: PointCloud2) -> np.ndarray:
        pts = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
        if len(pts) == 0:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array([[p[0], p[1], p[2]] for p in pts], dtype=np.float32)

    def cb_pc(self, msg: PointCloud2):
        if self.latest_mask is None or self.fx is None:
            # no mask or intrinsics yet
            return

        pts = self._pc2_to_xyz(msg)
        if pts.shape[0] == 0:
            self.pub.publish(create_pc2_xyz(np.empty((0, 3), np.float32), msg.header.frame_id, msg.header.stamp))
            return

        # cap points for speed (deterministic by nearest-to-origin)
        if self.max_points is not None and pts.shape[0] > self.max_points:
            d = np.linalg.norm(pts, axis=1)
            idx = np.argsort(d)[:self.max_points]
            pts = pts[idx]

        h, w = self.latest_mask.shape[:2]

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        # basic z filter (avoid division & garbage)
        valid_z = (z > self.min_z) & (z < self.max_z)
        if not np.any(valid_z):
            self.pub.publish(create_pc2_xyz(np.empty((0, 3), np.float32), msg.header.frame_id, msg.header.stamp))
            return

        x = x[valid_z]
        y = y[valid_z]
        z = z[valid_z]
        pts_v = pts[valid_z]

        # pinhole projection (assumes points are in the same camera optical frame as camera_info)
        u = (self.fx * (x / z) + self.cx).astype(np.int32)
        v = (self.fy * (y / z) + self.cy).astype(np.int32)

        in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        if not np.any(in_img):
            self.pub.publish(create_pc2_xyz(np.empty((0, 3), np.float32), msg.header.frame_id, msg.header.stamp))
            return

        u = u[in_img]
        v = v[in_img]
        pts_v = pts_v[in_img]

        mask_vals = self.latest_mask[v, u]
        keep = mask_vals >= self.mask_min_value

        filtered = pts_v[keep]
        self.pub.publish(create_pc2_xyz(filtered.astype(np.float32), msg.header.frame_id, msg.header.stamp))


def main(args=None):
    rclpy.init(args=args)
    node = RGBMotionPointCloudFilter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
