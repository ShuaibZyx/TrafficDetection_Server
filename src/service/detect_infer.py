from src.utils.image_utils import frame2image
import numpy as np
from torchvision.ops import box_convert
import torch


def postprocess(cls_out, box_out, ow, oh, conf_thresh, class_names):
    class_names = np.array(class_names)

    cls_out = torch.from_numpy(cls_out)
    box_out = torch.from_numpy(box_out)
    # process box_out
    bbox_pred = box_convert(box_out, in_fmt="cxcywh", out_fmt="xyxy")
    scale = torch.tensor([ow, oh, ow, oh], device=bbox_pred.device)
    bbox_pred *= scale

    # process cls_out
    scores = torch.sigmoid(cls_out)
    scores, labels = scores.max(-1)

    # 过滤置信度低于阈值的结果
    valid_mask = scores > conf_thresh
    labels = labels[valid_mask].numpy()
    boxes = bbox_pred[valid_mask].numpy()
    scores = scores[valid_mask].numpy()
    # 批量索引中文名
    names = class_names[labels]

    return dict(
        labels=labels.tolist(),
        boxes=boxes.tolist(),
        scores=scores.tolist(),
        names=list(names),
    )


def run_detection(session, frame, image_sizes, conf_thresh, class_names):
    # 帧图像高宽
    orig_h, orig_w = frame.shape[:2]
    # 转为模型可处理的image
    image_data = frame2image(frame, image_sizes)
    # 获取session的输入输出键名
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    # 执行推理
    outputs = session.run(output_names, {input_name: image_data})
    # 解构输出为类别和标注框
    cls_out, box_out = outputs[0], outputs[1]
    # 后处理为需要的输出结果
    result = postprocess(cls_out, box_out, orig_w, orig_h, conf_thresh, class_names)
    return result


