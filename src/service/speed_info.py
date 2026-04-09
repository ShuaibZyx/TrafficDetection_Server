from dataclasses import dataclass
import math
import cv2
import numpy as np
from collections import defaultdict

# ---------- 透视变换 ----------
SRC = np.array([[250, 809], [1545, 809], [1120, 262], [740, 262]], np.float32)
DST = np.array([[0, 24], [10.5, 24], [10.5, 0], [0, 0]], np.float32)
M = cv2.getPerspectiveTransform(SRC, DST)

# ---------- 经纬度 ----------
ORIGIN_LON, ORIGIN_LAT = 110.55434352378008, 21.22657328205872
METERS_PER_LON = 111320 * math.cos(math.radians(ORIGIN_LAT))
METERS_PER_LAT = 111320


@dataclass
class MotionFields:
    speed_x: float
    speed_y: float
    speed_heading: float
    acceleration_x: float
    acceleration_y: float
    acc_heading: float
    lontitude: float
    lantitude: float
    altitude: float
    heading: float


# ---------- 历史缓存 ----------
_track_history = defaultdict(list)


def compute_motion_fields(bbox, tid, timestamp):
    """
    bbox: (x, y, w, h)
    tid: int
    timestamp: ms
    """
    x, y, w, h = bbox
    cx, cy = x + w / 2, y + h / 2

    # 像素 → 世界
    pt = np.array([[[cx, cy]]], np.float32)
    wx, wy = cv2.perspectiveTransform(pt, M)[0, 0]

    # 经纬度
    lon = ORIGIN_LON - (wy - 24) / METERS_PER_LON
    lat = ORIGIN_LAT + wx / METERS_PER_LAT

    history = _track_history[tid]
    history.append((timestamp, wx, wy, w, h))

    # 默认值
    sx = sy = ax = ay = 0.0
    spd_head = acc_head = hdg = 0.0

    if len(history) >= 2:
        t0, x0, y0, _, _ = history[-2]
        t1, x1, y1, _, _ = history[-1]
        dt = (t1 - t0) / 1000.0

        if dt > 0:
            v = math.hypot(x1 - x0, y1 - y0) / dt
            hdg = math.atan2(y1 - y0, x1 - x0)
            sx = v * math.cos(hdg)
            sy = v * math.sin(hdg)
            spd_head = hdg

    if len(history) >= 3:
        t0, x0, y0, _, _ = history[-3]
        t1, x1, y1, _, _ = history[-2]
        t2, x2, y2, _, _ = history[-1]

        dt1 = (t1 - t0) / 1000.0
        dt2 = (t2 - t1) / 1000.0

        if dt1 > 0 and dt2 > 0:
            v1 = math.hypot(x1 - x0, y1 - y0) / dt1
            v2 = math.hypot(x2 - x1, y2 - y1) / dt2
            a = (v2 - v1) / dt2
            acc_head = math.atan2(y2 - y1, x2 - x1)
            ax = a * math.cos(acc_head)
            ay = a * math.sin(acc_head)

    return MotionFields(
        speed_x=float(sx),
        speed_y=float(sy),
        speed_heading=float(spd_head),
        acceleration_x=float(ax),
        acceleration_y=float(ay),
        acc_heading=float(acc_head),
        lontitude=float(lon),
        lantitude=float(lat),
        altitude=float(h / 2),
        heading=float(hdg),
    )
