import numpy as np

import onnxruntime as ort
from os.path import exists, basename

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def load_onnx_model(path, providers):
    if exists(path):
        print(f"加载模型: {basename(path)}")
        return ort.InferenceSession(path, providers=providers)
    else:
        print(f"警告：找不到模型文件 {path}")
        return None
