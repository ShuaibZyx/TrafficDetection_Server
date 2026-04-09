from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from src.service.detect_infer import run_detection
from src.service.tracker_adapter import tracker_infer
from src.utils.video_utils import VideoReader
import json
from fastapi import APIRouter, UploadFile, File, Request
from os.path import join, basename
from src.template.response import response_success
from pathlib import Path

router = APIRouter()

@router.get("/sse/second/{video_id}")
async def video_infer_sse(request: Request, video_id: str):
    base_dir = request.app.state.base_dir
    model = request.app.state.detect_model
    config = request.app.state.config

    upload_dir = join(config.app.static, config.detect.upload_dir)
    sample_rate = config.detect.sample_rate

    video_path = join(base_dir, upload_dir, video_id)
    reader = VideoReader(video_path, sample_rate=sample_rate)
    tracker = request.app.state.tracker
    tid_cache = {}

    image_sizes = config.detect.image_sizes
    conf_thresh = config.detect.conf_thresh
    class_names = config.detect.class_names

    fps = reader.fps // sample_rate

    async def event_generator():
        try:
            second_result = []
            for frame, idx in reader:
                if frame is None:
                    break
                # 推理
                detect_result = run_detection( model, frame, image_sizes, conf_thresh, class_names)
                track_result = tracker_infer(
                    tracker, detect_result, image_sizes, class_names, tid_cache
                )
                second_result.append(
                    {"idx": idx, "second": round(idx / fps, 6), **track_result}
                )

                # 一秒发送一次
                if len(second_result) >= fps:
                    yield f"data: {json.dumps(second_result, ensure_ascii=False)}\n\n"
                    second_result.clear()

                if await request.is_disconnected():
                    break

            if len(second_result) > 0:
                yield f"data: {json.dumps(second_result, ensure_ascii=False)}\n"

        finally:
            reader.release()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.post("/api/upload/video")
async def upload_video(request: Request, file: UploadFile = File(...)):
    base_dir = request.app.state.base_dir
    config = request.app.state.config

    filename = basename(file.filename)  # 防止路径穿越
    file_path = join(base_dir, config.app.static, config.detect.upload_dir, filename)

    content = await file.read()
    with open(file_path, "wb") as f:
        f.write(content)

    return response_success(data=filename)


@router.get("/api/list/video")
async def list_videos(request: Request):
    base_dir = request.app.state.base_dir
    config = request.app.state.config

    upload_dir = join(base_dir, config.app.static, config.detect.upload_dir)
    files = Path(upload_dir).glob("*.mp4")
    names = [f.name for f in files]
    return response_success(data=names)
