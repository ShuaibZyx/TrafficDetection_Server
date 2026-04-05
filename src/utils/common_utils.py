import onnxruntime as ort
from os.path import exists, basename


def load_onnx_model(path, providers):
    if exists(path):
        print(f"加载模型: {basename(path)}")
        return ort.InferenceSession(path, providers=providers)
    else:
        print(f"警告：找不到模型文件 {path}")
        return None
