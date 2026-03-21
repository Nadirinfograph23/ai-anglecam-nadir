import asyncio
import base64
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from app.config import (
    MAX_UPLOAD_SIZE_MB,
    PREDEFINED_ANGLES,
)
from app.services.hf_client import (
    clamp_rotate,
    compute_image_hash,
    convert_forward,
    convert_vertical,
    hf_client,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting up AI AngleCam Nadir backend...")
    yield
    logger.info("Shutting down...")
    await hf_client.close()


app = FastAPI(title="AI AngleCam Nadir", lifespan=lifespan)

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


class GenerateAllResponse(BaseModel):
    results: list[AngleResult]
    total: int
    successful: int
    failed: int


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/api/angles")
async def get_angles():
    """Return the list of predefined camera angles."""
    return {"angles": PREDEFINED_ANGLES}


@app.get("/api/status")
async def get_status():
    """Return current system status including cache, queue, and token info."""
    return hf_client.get_status()


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
        image_hash = compute_image_hash(image_data)
        optimized = hf_client.optimize_image(image_data)
        uploaded_path = await hf_client.upload_image(optimized)

        rotate = clamp_rotate(rotate_deg)
        img_bytes, content_type = await hf_client.generate_angle(
            uploaded_path=uploaded_path,
            image_hash=image_hash,
            rotate_deg=rotate,
            move_forward=move_forward,
            vertical_tilt=vertical_tilt,
            wideangle=wideangle,
        )

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return JSONResponse({
            "success": True,
            "image_data": b64,
            "content_type": content_type,
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
        results = await hf_client.generate_all_angles(image_data, lens=lens)
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
        image_hash = compute_image_hash(image_data)
        optimized = hf_client.optimize_image(image_data)

        try:
            uploaded_path = await hf_client.upload_image(optimized)
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': f'Upload failed: {str(e)}'})}\n\n"
            return

        forward = convert_forward(lens)
        completed = 0
        total = len(PREDEFINED_ANGLES)

        yield f"data: {json.dumps({'type': 'start', 'total': total})}\n\n"

        # Create all tasks with angle name tracking
        pending_tasks = {}
        for angle in PREDEFINED_ANGLES:
            rotate = clamp_rotate(float(angle["h"]))
            tilt = convert_vertical(float(angle["v"]))
            task = asyncio.create_task(
                hf_client.generate_angle(
                    uploaded_path=uploaded_path,
                    image_hash=image_hash,
                    rotate_deg=rotate,
                    move_forward=forward,
                    vertical_tilt=tilt,
                    wideangle=(lens == "wide"),
                )
            )
            pending_tasks[task] = angle["name"]

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
                    img_bytes, content_type = task.result()
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    yield f"data: {json.dumps({'type': 'result', 'name': angle_name, 'success': True, 'image_data': b64, 'content_type': content_type, 'completed': completed, 'total': total})}\n\n"
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
    image_hash = compute_image_hash(image_data)
    optimized = hf_client.optimize_image(image_data)

    try:
        uploaded_path = await hf_client.upload_image(optimized)
        forward = convert_forward(lens)
        rotate = clamp_rotate(float(angle_config["h"]))
        tilt = convert_vertical(float(angle_config["v"]))

        img_bytes, content_type = await hf_client.generate_angle(
            uploaded_path=uploaded_path,
            image_hash=image_hash,
            rotate_deg=rotate,
            move_forward=forward,
            vertical_tilt=tilt,
            wideangle=(lens == "wide"),
        )

        b64 = base64.b64encode(img_bytes).decode("utf-8")
        return JSONResponse({
            "name": angle_name,
            "success": True,
            "image_data": b64,
            "content_type": content_type,
        })
    except Exception as e:
        logger.error("Retry for %s failed: %s", angle_name, str(e))
        raise HTTPException(status_code=500, detail=str(e))
