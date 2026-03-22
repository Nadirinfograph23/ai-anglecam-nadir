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
from app.services.provider_manager import (
    compute_image_hash,
    optimize_image,
    provider_manager,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(application: FastAPI):
    logger.info("Starting up AI AngleCam Nadir backend...")
    logger.info("Active providers: %s", provider_manager.get_active_providers())
    yield
    logger.info("Shutting down...")
    await provider_manager.close()


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
    return {
        "status": "ok",
        "providers": provider_manager.get_active_providers(),
    }


@app.get("/api/angles")
async def get_angles():
    """Return the list of predefined camera angles."""
    return {"angles": PREDEFINED_ANGLES}


@app.post("/api/generate-single")
async def generate_single(
    image: UploadFile = File(...),
    h_angle: float = Form(0.0),
    v_angle: float = Form(0.0),
    lens: str = Form("normal"),
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
        optimized = optimize_image(image_data)

        img_bytes, content_type = await provider_manager.generate_angle(
            image_data=optimized,
            image_hash=image_hash,
            h_angle=h_angle,
            v_angle=v_angle,
            lens=lens,
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

    Uses multi-provider fallback with parallel processing,
    controlled concurrency, automatic retries, and caching.
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
        results: list[dict] = []
        async for item in provider_manager.generate_all_angles_stream(image_data, lens=lens):
            if item["type"] == "result":
                results.append(item)

        successful = sum(1 for r in results if r.get("success"))
        failed = len(results) - successful

        return GenerateAllResponse(
            results=[AngleResult(
                name=r["name"],
                success=r["success"],
                image_data=r.get("image_data"),
                content_type=r.get("content_type"),
                error=r.get("error"),
            ) for r in results],
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

    Uses multi-provider fallback. Streams results as they complete,
    so the frontend can show images progressively.
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
        total = len(PREDEFINED_ANGLES)
        providers = provider_manager.get_active_providers()

        yield f"data: {json.dumps({'type': 'start', 'total': total, 'providers': providers})}\n\n"

        last_completed = 0
        try:
            async for item in provider_manager.generate_all_angles_stream(image_data, lens=lens):
                yield f"data: {json.dumps(item)}\n\n"
                if item.get("completed"):
                    last_completed = item["completed"]
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            return

        yield f"data: {json.dumps({'type': 'done', 'completed': last_completed, 'total': total})}\n\n"

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
    """Retry generating a specific failed angle using multi-provider fallback."""
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
    optimized = optimize_image(image_data)

    try:
        img_bytes, content_type = await provider_manager.generate_angle(
            image_data=optimized,
            image_hash=image_hash,
            h_angle=float(angle_config["h"]),
            v_angle=float(angle_config["v"]),
            lens=lens,
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


@app.get("/api/providers")
async def get_providers():
    """Return the list of active providers."""
    return {"providers": provider_manager.get_active_providers()}
