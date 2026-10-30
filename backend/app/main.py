import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router, separator


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # load the model in the background so the server answers straight away
    threading.Thread(target=separator.warm_up, daemon=True).start()
    yield


app = FastAPI(title="riffapart API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://riff-apart.vercel.app",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok", "device": separator.device.type}
