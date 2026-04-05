from fastapi import FastAPI
from src.api.video import router as video_router
from src.core.lifespan import lifespan
from src.core.middleware import setup_cors, setup_static
from src.template.exception import register_global_exception
from src.config_loader import load_config
import os
import uvicorn

app = FastAPI(lifespan=lifespan)

config = load_config()
base_dir = os.getcwd()
app.state.config = config
app.state.base_dir = base_dir

setup_cors(app)
setup_static(app)
register_global_exception(app)
app.include_router(video_router)

def main():
    uvicorn.run(app, host=config.app.host, port=config.app.port, reload=False)

if __name__ == "__main__":
    main()
