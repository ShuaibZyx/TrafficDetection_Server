from contextlib import asynccontextmanager
from os.path import join
from src.utils.common_utils import load_onnx_model
from pathlib import Path
import onnxruntime as ort
from loguru import logger

BASE_DIR = Path(__file__).resolve().parents[2]

@asynccontextmanager
async def lifespan(app):

    providers = ort.get_available_providers()
    logger.info(f"可用设备: {providers}")

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    detect_model_path = join(BASE_DIR, "weight_file", "detection_R39.onnx")
    app.state.detect_model = load_onnx_model(detect_model_path, providers)

    yield
    if hasattr(app.state, "detect_model"):
        del app.state.detect_model

    logger.info("模型资源已释放")
