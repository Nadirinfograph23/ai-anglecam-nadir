import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.config import (
    MAX_UPLOAD_SIZE_MB,
    PREDEFINED_ANGLES,
    STAGGER_DELAY,
)
from app.services.hf_client import (
    clamp_rotate,
    convert_forward,
    convert_vertical,
)
from app.services.provider_manager import provider_manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting up AI AngleCam Nadir backend...")
    yield
    logger.info("Shutting down...")
    await provider_manager.close()


app = FastAPI(title="AI AngleCam Nadir", lifespan=lifespan)

# GZip compression for faster response delivery
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Disable CORS. Do not remove this for full-stack development.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class AngleResult(BaseModel):
    name: str
    success: bool
    image_data: str | None = None
    content_type: str | None = None
    error: str | None = None
    provider: str | None = None


class GenerateAllResponse(BaseModel):
    results: list[AngleResult]
    total: int
    successful: int
    failed: int


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/api/provider-status")
async def get_provider_status():
    """Return current health status of all generation providers."""
    return provider_manager.get_provider_status()


@app.get("/api/angles")
async def get_angles():
    """Return the list of predefined camera angles."""
    return {"angles": PREDEFINED_ANGLES}


@app.post("/api/generate-single")
async def generate_single(
    image: UploadFile = File(...),
    rotate_deg: float = Form(0.0),
    move_forward: float = Form(2.0),
    vertical_tilt: float = Form(0.0),
    wideangle: bool = Form(False),
):
    """Generate a single angle image from the uploaded image."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_data = await image.read()
    if len(image_data) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large. Max size is {MAX_UPLOAD_SIZE_MB}MB"
        )

    try:
        rotate = clamp_rotate(rotate_deg)
        img_bytes, content_type, provider = await provider_manager.generate_angle(
            image_data=image_data,
            rotate_deg=rotate,
            move_forward=move_forward,
            vertical_tilt=vertical_tilt,
            wideangle=wideangle,
            v_raw=vertical_tilt * 60.0,
        )

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return JSONResponse({
            "success": True,
            "image_data": b64,
            "content_type": content_type,
            "provider": provider,
        })
    except Exception as e:
        logger.error("Single generation failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-all", response_model=GenerateAllResponse)
async def generate_all(
    image: UploadFile = File(...),
    lens: str = Form("normal"),
):
    """Generate all 9 camera angle images from the uploaded image.

    Uses parallel processing with controlled concurrency, automatic retries,
    and caching for reliability.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_data = await image.read()
    if len(image_data) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large. Max size is {MAX_UPLOAD_SIZE_MB}MB"
        )

    try:
        results = await provider_manager.generate_all_angles(image_data, lens=lens)
        successful = sum(1 for r in results if r.get("success"))
        failed = len(results) - successful

        return GenerateAllResponse(
            results=[AngleResult(**r) for r in results],
            total=len(results),
            successful=successful,
            failed=failed,
        )
    except Exception as e:
        logger.error("All-angles generation failed: %s", str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-stream")
async def generate_stream(
    image: UploadFile = File(...),
    lens: str = Form("normal"),
):
    """Generate all 9 angles with SSE streaming progress updates.

    Streams results as they complete, so the frontend can show
    images progressively instead of waiting for all 9.
    """
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_data = await image.read()
    if len(image_data) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large. Max size is {MAX_UPLOAD_SIZE_MB}MB"
        )

    async def event_stream():
        forward = convert_forward(lens)
        completed = 0
        total = len(PREDEFINED_ANGLES)

        yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

        # Create tasks with staggered launches to reduce API pressure
        pending_tasks = {}
        for i, angle in enumerate(PREDEFINED_ANGLES):
            if i > 0:
                await asyncio.sleep(STAGGER_DELAY)
            rotate = clamp_rotate(float(angle["h"]))
            tilt = convert_vertical(float(angle["v"]))
            task = asyncio.create_task(
                provider_manager.generate_angle(
                    image_data=image_data,
                    rotate_deg=rotate,
                    move_forward=forward,
                    vertical_tilt=tilt,
                    wideangle=(lens == "wide"),
                    v_raw=float(angle["v"]),
                )
            )
            pending_tasks[task] = angle["name"]

            # Yield any results that completed while we were staggering
            done_early = [t for t in pending_tasks if t.done()]
            for task_done in done_early:
                angle_name = pending_tasks.pop(task_done)
                completed += 1
                try:
                    img_bytes, content_type, provider = task_done.result()
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    yield f"data: {json.dumps({'type': 'result', 'name': angle_name, 'success': True, 'image_data': b64, 'content_type': content_type, 'provider': provider, 'completed': completed, 'total': total})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'result', 'name': angle_name, 'success': False, 'error': str(e), 'completed': completed, 'total': total})}\n\n"

        # Yield results as they complete
        while pending_tasks:
            done, _ = await asyncio.wait(
                pending_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                angle_name = pending_tasks.pop(task)
                completed += 1
                try:
                    img_bytes, content_type, provider = task.result()
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    yield f"data: {json.dumps({'type': 'result', 'name': angle_name, 'success': True, 'image_data': b64, 'content_type': content_type, 'provider': provider, 'completed': completed, 'total': total})}\n\n"
                except Exception as e:
                    yield f"data: {json.dumps({'type': 'result', 'name': angle_name, 'success': False, 'error': str(e), 'completed': completed, 'total': total})}\n\n"

        yield f"data: {json.dumps({'type': 'done', 'completed': completed, 'total': total})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/retry-angle")
async def retry_angle(
    image: UploadFile = File(...),
    angle_name: str = Form(...),
    lens: str = Form("normal"),
):
    """Retry generating a specific failed angle."""
    if not image.content_type or not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    angle_config = None
    for angle in PREDEFINED_ANGLES:
        if angle["name"] == angle_name:
            angle_config = angle
            break

    if angle_config is None:
        raise HTTPException(status_code=400, detail=f"Unknown angle: {angle_name}")

    image_data = await image.read()

    try:
        forward = convert_forward(lens)
        rotate = clamp_rotate(float(angle_config["h"]))
        tilt = convert_vertical(float(angle_config["v"]))

        img_bytes, content_type, provider = await provider_manager.generate_angle(
            image_data=image_data,
            rotate_deg=rotate,
            move_forward=forward,
            vertical_tilt=tilt,
            wideangle=(lens == "wide"),
            v_raw=float(angle_config["v"]),
        )

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return JSONResponse({
            "name": angle_name,
            "success": True,
            "image_data": b64,
            "content_type": content_type,
            "provider": provider,
        })
    except Exception as e:
        logger.error("Retry for %s failed: %s", angle_name, str(e))
        raise HTTPException(status_code=500, detail=str(e))
