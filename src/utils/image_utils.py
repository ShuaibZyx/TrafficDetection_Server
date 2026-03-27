import cv2
import numpy as np
import base64


def frame2image(frame, image_sizes=(640, 640)):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, image_sizes)
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image


def box2image(x, y, w, h, ow, oh):
    rw = w * ow
    rh = h * oh
    cx = x * ow
    cy = y * oh
    x_min = cx - rw / 2
    y_min = cy - rh / 2
    return x_min, y_min, rw, rh


def encode_image_base64(
    frame,
    quality=80,
    optimize=True,
    progressive=False,
):
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    if optimize:
        params += [int(cv2.IMWRITE_JPEG_OPTIMIZE), 1]
    if progressive:
        params += [int(cv2.IMWRITE_JPEG_PROGRESSIVE), 1]
    
    _, buffer = cv2.imencode(".jpg", frame, params)
    image_base64 = base64.b64encode(buffer).decode()
    return image_base64
