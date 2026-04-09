import numpy as np
from src.service.attach_info import compute_attributes
from src.service.speed_info import compute_motion_fields


def detect_to_tracker(detect_result):
    boxes = np.asarray(detect_result["boxes"], dtype=np.float64)
    scores = np.asarray(detect_result["scores"], dtype=np.float64)
    labels = np.asarray(detect_result["labels"], dtype=np.float64)

    tracker_input = np.hstack(
        [
            boxes[:, :4],
            np.ones((boxes.shape[0], 1), dtype=np.float64),
            scores[:, None],
            labels[:, None],
        ]
    )

    return tracker_input


def tracker_infer(tracker, detect_result, image_sizes, class_names, tid_cache):
    h, w = image_sizes[:2]
    tracker_input = detect_to_tracker(detect_result)
    tracks = tracker.update(
        tracker_input,
        img_info=[h, w],
        img_size=(h, w),
    )
    result = tracker_to_frontend(tracks, class_names, h, tid_cache)
    return result


def tracker_to_frontend(tracks, class_names, img_h, tid_cache):
    payload = dict(
        track_ids=[], #追踪id
        labels=[], # 类别编号
        boxes=[], # bbox框
        scores=[], # 置信度得分
        names=[], # string 类别名称
        security_level=[],  # number 安全等级
        length=[],  # number[] // 长
        width=[],  # number[] // 宽
        height=[],  # number[] // 高
        speed_x=[],  # number[] // 纵向速度
        speed_y=[],  # number[] // 横向速度
        speed_heading=[],  # number[] // 速度方向
        acceleration_x=[],  # number[] // 纵向加速度
        acceleration_y=[],  # number[] // 横向加速度
        acc_heading=[],  # number[] // 加速度方向
        lontitude=[],  # number[] // 经度
        lantitude=[],  # number[] // 纬度
        altitude=[],  # number[] // 海拔
        heading=[],  # number[] // 运动方向
        plate=[],  # string[] // 车牌
    )

    for t in tracks:
        tlwh = np.asarray(t.tlwh, dtype=np.float64)
        if tlwh.size < 4:
            continue

        x, y, w, h = np.round(tlwh[:4], 2)
        label = int(t.classid)

        tid = int(t.track_id)
        score = round(float(t.score), 4)
        box = [x, y, round(x + w, 2), round(y + h, 2)]
        name = class_names[label]

        # attributes
        security, size = compute_attributes(
            (x, y, w, h), tid, label + 1, img_h, tid_cache
        )
        length, width, height = size

        motion = compute_motion_fields((x, y, w, h), tid, t.frame_id)

        payload["track_ids"].append(tid)
        payload["labels"].append(label)
        payload["boxes"].append(box)
        payload["scores"].append(score)
        payload["names"].append(name)
        payload["security_level"].append(security)
        payload["length"].append(length)
        payload["width"].append(width)
        payload["height"].append(height)

        payload["speed_x"].append(motion.speed_x)
        payload["speed_y"].append(motion.speed_y)
        payload["speed_heading"].append(motion.speed_heading)
        payload["acceleration_x"].append(motion.acceleration_x)
        payload["acceleration_y"].append(motion.acceleration_y)
        payload["acc_heading"].append(motion.acc_heading)
        payload["lontitude"].append(motion.lontitude)
        payload["lantitude"].append(motion.lantitude)
        payload["altitude"].append(motion.altitude)
        payload["heading"].append(motion.heading)

    return payload
