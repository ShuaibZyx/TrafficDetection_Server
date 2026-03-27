from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.service.detect_infer import run_detection
from src.utils.video_utils import VideoReader
from src.utils.image_utils import encode_image_base64
import json
import uuid
from fastapi import APIRouter, UploadFile, File, Request
from os.path import join
from loguru import logger
from src.template.response import response_success
from pathlib import Path

router = APIRouter()

@router.websocket("/ws/frame/{video_id}")
async def infer_video(websocket: WebSocket, video_id: str):
    await websocket.accept()

    base_dir = websocket.app.state.base_dir
    model = websocket.app.state.detect_model
    config = websocket.app.state.config

    video_path = join(base_dir, config.detect.upload_dir, video_id)
    reader = VideoReader(video_path, sample_rate=1)

    image_sizes = config.detect.image_sizes
    conf_thresh = config.detect.conf_thresh
    class_names = config.detect.class_names

    try:
        for frame in reader:
            # 模型推理，得到图像级别的标注框结果
            image_boxes = run_detection(
                model, frame, image_sizes, conf_thresh, class_names
            )
            # 帧图像转为base64
            image_base64 = encode_image_base64(frame, quality=80)
            # json序列化
            result_json = json.dumps({"image": image_base64, "boxes": image_boxes})
            # 发送结果
            await websocket.send_text(result_json)

    except WebSocketDisconnect:
        logger.info(f"客户端断开: {video_id}")

    except Exception as e:
        logger.error(f"推理异常: {e}")

    finally:
        reader.release()


@router.post("/api/upload/video")
async def upload_video(request: Request, file: UploadFile = File(...)):
    video_id = f"{uuid.uuid4().hex}.mp4"
    base_dir = request.app.state.base_dir
    config = request.app.state.config

    file_path = join(base_dir, config.detect.upload_dir, video_id)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return response_success(data=video_id)


# ================= 获取视频列表 =================
@router.get("/api/list/video")
async def list_videos(
    request: Request
):
    base_dir = request.app.state.base_dir
    config = request.app.state.config

    upload_dir = join(base_dir, config.detect.upload_dir)
    files = Path(upload_dir).glob("*.mp4")
    names = [f.name for f in files]
    return response_success(data=names)
