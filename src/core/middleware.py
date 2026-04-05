from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from os.path import join

def setup_cors(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def setup_static(app):
    config = app.state.config
    base_dir = app.state.base_dir
    directory = join(base_dir, config.app.static, config.detect.upload_dir)
    app.mount("/static/videos", StaticFiles(directory=directory), name="videos")
