import cv2
from loguru import logger
from os.path import basename
from loguru import logger

class VideoReader:
    # sample_rate: 抽帧率（每 N 帧取 1 帧）
    def __init__(self, video_path, sample_rate=1):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise RuntimeError("Failed to open video")

        self.fps = round(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        logger.info(f"视频 {basename(video_path)} 原始 FPS 为: {self.fps}")

        self.sample_rate = sample_rate
        self.idx = 0

    def __iter__(self):
        return self

    def __next__(self):
        while True:
            ret = self.cap.grab()
            if not ret:
                self.release()

            self.idx += 1

            # 抽帧
            if self.idx % self.sample_rate != 0:
                continue

            ret, frame = self.cap.retrieve()
            if not ret or frame is None:
                self.release()
            return frame, self.idx

    def release(self):
        if self.cap:
            self.cap.release()
