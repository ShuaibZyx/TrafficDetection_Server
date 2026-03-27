from src.utils.image_utils import frame2image, box2image
from src.utils.common_utils import softmax
import numpy as np


def postprocess(cls_out, box_out, ow, oh, conf_thresh, class_names):
    cls_out, box_out = cls_out[0], box_out[0]
    scores = softmax(cls_out)

    results = []
    for i in range(box_out.shape[0]):
        x, y, w, h = box_out[i]
        bx, by, bw, bh = box2image(x, y, w, h, ow, oh)

        label = np.argmax(scores[i])
        conf = scores[i][label]

        if conf > conf_thresh:
            results.append(
                {
                    "x": float(bx),
                    "y": float(by),
                    "width": float(bw),
                    "height": float(bh),
                    "confidence": round(float(conf), 2),
                    "type": class_names[int(label)],
                }
            )
    return results


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
