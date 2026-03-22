"""HuggingFace Gradio Space client with retries, caching, and concurrency control."""

import asyncio
import base64
import hashlib
import io
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

import httpx
from PIL import Image

from app.config import (
    CACHE_MAX_SIZE,
    CACHE_TTL_SECONDS,
    DEFAULT_GUIDANCE_SCALE,
    DEFAULT_HEIGHT,
    DEFAULT_INFERENCE_STEPS,
    DEFAULT_WIDTH,
    HF_API_TOKEN,
    HF_SPACE_URL,
    MAX_CONCURRENT_GENERATIONS,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
)

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    data: bytes
    content_type: str
    created_at: float = field(default_factory=time.time)


class ImageCache:
    """LRU cache for generated images with TTL."""

    def __init__(self, max_size: int = CACHE_MAX_SIZE, ttl: int = CACHE_TTL_SECONDS):
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl

    def _make_key(self, image_hash: str, rotate: float, forward: float, tilt: float, wide: bool) -> str:
        return f"{image_hash}:{rotate}:{forward}:{tilt}:{wide}"

    def get(self, image_hash: str, rotate: float, forward: float, tilt: float, wide: bool) -> CacheEntry | None:
        key = self._make_key(image_hash, rotate, forward, tilt, wide)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() - entry.created_at > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return entry

    def put(self, image_hash: str, rotate: float, forward: float, tilt: float, wide: bool,
            data: bytes, content_type: str) -> None:
        key = self._make_key(image_hash, rotate, forward, tilt, wide)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = CacheEntry(data=data, content_type=content_type)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


def compute_image_hash(image_data: bytes) -> str:
    return hashlib.sha256(image_data).hexdigest()[:16]


def clamp_rotate(deg: float) -> float:
    if -90 <= deg <= 90:
        return deg
    if 90 < deg <= 180:
        return 90.0
    if -180 <= deg < -90:
        return -90.0
    return 0.0


def convert_vertical(v: float) -> float:
    return max(-1.0, min(1.0, v / 60.0))


def convert_forward(lens: str) -> float:
    mapping = {"closeup": 5.0, "wide": 0.0, "normal": 2.0}
    return mapping.get(lens, 2.0)


class HFClient:
    """Client for interacting with the HuggingFace Gradio Space API."""

    def __init__(self) -> None:
        self._cache = ImageCache()
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
        self._http_client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "HuggingFace Space"

    @property
    def is_available(self) -> bool:
        return True  # HF Space can work without token for public spaces

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=30.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._http_client

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def upload_image(self, image_data: bytes, filename: str = "input.png") -> str:
        """Upload image to HF Space and return the file path."""
        client = await self._get_client()
        token = HF_API_TOKEN

        for attempt in range(MAX_RETRIES):
            try:
                files = {"files": (filename, image_data, "image/png")}
                response = await client.post(
                    f"{HF_SPACE_URL}/gradio_api/upload",
                    files=files,
                    headers={"Authorization": f"Bearer {token}"},
                )
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0]
                    raise ValueError(f"Unexpected upload response: {result}")

                logger.warning(
                    "Upload attempt %d failed: %d %s",
                    attempt + 1, response.status_code, response.text[:200]
                )
            except httpx.TimeoutException:
                logger.warning("Upload attempt %d timed out", attempt + 1)
            except Exception as e:
                logger.warning("Upload attempt %d error: %s", attempt + 1, str(e))

            if attempt < MAX_RETRIES - 1:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                await asyncio.sleep(delay)

        raise RuntimeError("Failed to upload image after all retries")

    async def generate_angle(
        self,
        uploaded_path: str,
        image_hash: str,
        rotate_deg: float,
        move_forward: float,
        vertical_tilt: float,
        wideangle: bool,
        seed: int = 0,
        randomize_seed: bool = True,
    ) -> tuple[bytes, str]:
        """Generate a single angle image. Returns (image_bytes, content_type)."""
        # Check cache first
        cached = self._cache.get(image_hash, rotate_deg, move_forward, vertical_tilt, wideangle)
        if cached is not None:
            logger.info("Cache hit for %s rotate=%.1f fwd=%.1f tilt=%.1f wide=%s",
                        image_hash, rotate_deg, move_forward, vertical_tilt, wideangle)
            return cached.data, cached.content_type

        # Acquire semaphore for concurrency control
        async with self._semaphore:
            return await self._generate_with_retry(
                uploaded_path, image_hash, rotate_deg, move_forward,
                vertical_tilt, wideangle, seed, randomize_seed
            )

    async def _generate_with_retry(
        self,
        uploaded_path: str,
        image_hash: str,
        rotate_deg: float,
        move_forward: float,
        vertical_tilt: float,
        wideangle: bool,
        seed: int,
        randomize_seed: bool,
    ) -> tuple[bytes, str]:
        """Generate with retry logic."""
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                result = await self._call_gradio_api(
                    uploaded_path, rotate_deg, move_forward,
                    vertical_tilt, wideangle, seed, randomize_seed
                )
                # Cache the result
                self._cache.put(
                    image_hash, rotate_deg, move_forward, vertical_tilt, wideangle,
                    result[0], result[1]
                )
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    "Generation attempt %d/%d failed for rotate=%.1f: %s",
                    attempt + 1, MAX_RETRIES, rotate_deg, str(e)
                )
                if attempt < MAX_RETRIES - 1:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"Generation failed after {MAX_RETRIES} attempts: {last_error}"
        )

    async def _call_gradio_api(
        self,
        uploaded_path: str,
        rotate_deg: float,
        move_forward: float,
        vertical_tilt: float,
        wideangle: bool,
        seed: int,
        randomize_seed: bool,
    ) -> tuple[bytes, str]:
        """Make the actual Gradio API call."""
        client = await self._get_client()
        token = HF_API_TOKEN

        # Step 1: Submit the job
        payload = {
            "data": [
                False,  # is_reset_val
                {"path": uploaded_path, "meta": {"_type": "gradio.FileData"}},
                rotate_deg,
                move_forward,
                vertical_tilt,
                wideangle,
                seed,
                randomize_seed,
                DEFAULT_GUIDANCE_SCALE,
                DEFAULT_INFERENCE_STEPS,
                DEFAULT_WIDTH,
                DEFAULT_HEIGHT,
                None,  # prev_output
            ]
        }

        submit_response = await client.post(
            f"{HF_SPACE_URL}/gradio_api/call/maybe_infer",
            json=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
        )

        if submit_response.status_code != 200:
            raise RuntimeError(
                f"Submit failed: {submit_response.status_code} - {submit_response.text[:300]}"
            )

        event_id = submit_response.json().get("event_id")
        if not event_id:
            raise RuntimeError("No event_id in submit response")

        # Step 2: Poll for result (SSE stream)
        result_response = await client.get(
            f"{HF_SPACE_URL}/gradio_api/call/maybe_infer/{event_id}",
            headers={"Authorization": f"Bearer {token}"},
        )

        if result_response.status_code != 200:
            raise RuntimeError(
                f"Result fetch failed: {result_response.status_code}"
            )

        # Parse SSE response
        image_url = self._parse_sse_response(result_response.text)

        # Step 3: Download the generated image
        image_response = await client.get(
            image_url,
            headers={"Authorization": f"Bearer {token}"},
        )

        if image_response.status_code != 200:
            raise RuntimeError(
                f"Image download failed: {image_response.status_code}"
            )

        content_type = image_response.headers.get("content-type", "image/webp")
        return image_response.content, content_type

    def _parse_sse_response(self, text: str) -> str:
        """Parse SSE response to extract image URL."""
        lines = text.split("\n")
        error_msg = ""

        for line in lines:
            if line.startswith("event: error"):
                error_msg = "API returned an error"
            if line.startswith("data: "):
                data_str = line[6:].strip()
                if data_str == "null":
                    continue
                try:
                    data = json.loads(data_str)
                    if isinstance(data, list) and len(data) > 0:
                        first = data[0]
                        if isinstance(first, dict) and "url" in first:
                            return first["url"]
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue

        raise RuntimeError(error_msg or "No result image found in SSE response")

    async def generate_all_angles(
        self,
        image_data: bytes,
        lens: str = "normal",
        on_progress: asyncio.Queue | None = None,
    ) -> list[dict]:
        """Generate all 9 angle images with controlled parallelism."""
        from app.config import PREDEFINED_ANGLES

        image_hash = compute_image_hash(image_data)

        # Optimize image for upload
        optimized = self.optimize_image(image_data)

        # Upload once, reuse for all angles
        uploaded_path = await self.upload_image(optimized)

        forward = convert_forward(lens)

        # Create tasks for all angles
        results: list[dict] = []
        tasks = []

        for angle in PREDEFINED_ANGLES:
            rotate = clamp_rotate(float(angle["h"]))
            tilt = convert_vertical(float(angle["v"]))

            task = self._generate_angle_task(
                uploaded_path=uploaded_path,
                image_hash=image_hash,
                angle_name=angle["name"],
                rotate_deg=rotate,
                move_forward=forward,
                vertical_tilt=tilt,
                wideangle=(lens == "wide"),
                progress_queue=on_progress,
            )
            tasks.append(task)

        # Run with controlled parallelism (semaphore in generate_angle)
        task_results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(task_results):
            angle = PREDEFINED_ANGLES[i]
            if isinstance(result, Exception):
                results.append({
                    "name": angle["name"],
                    "success": False,
                    "error": str(result),
                })
            else:
                img_bytes, content_type = result
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                results.append({
                    "name": angle["name"],
                    "success": True,
                    "image_data": b64,
                    "content_type": content_type,
                })

        return results

    async def _generate_angle_task(
        self,
        uploaded_path: str,
        image_hash: str,
        angle_name: str,
        rotate_deg: float,
        move_forward: float,
        vertical_tilt: float,
        wideangle: bool,
        progress_queue: asyncio.Queue | None = None,
    ) -> tuple[bytes, str]:
        """Generate a single angle and report progress."""
        try:
            result = await self.generate_angle(
                uploaded_path=uploaded_path,
                image_hash=image_hash,
                rotate_deg=rotate_deg,
                move_forward=move_forward,
                vertical_tilt=vertical_tilt,
                wideangle=wideangle,
            )
            if progress_queue:
                await progress_queue.put({"name": angle_name, "status": "completed"})
            return result
        except Exception as e:
            if progress_queue:
                await progress_queue.put({"name": angle_name, "status": "failed", "error": str(e)})
            raise

    async def generate_angle_direct(
        self,
        image_data: bytes,
        h_angle: float,
        v_angle: float,
        lens: str = "normal",
    ) -> tuple[bytes, str]:
        """Generate a single angle from raw image data (no pre-upload needed).

        This method handles upload + generation in one call,
        used by the provider manager.
        """
        optimized = self.optimize_image(image_data)
        image_hash = compute_image_hash(image_data)
        uploaded_path = await self.upload_image(optimized)

        rotate = clamp_rotate(float(h_angle))
        forward = convert_forward(lens)
        tilt = convert_vertical(float(v_angle))

        # No semaphore here — the provider_manager already controls concurrency
        return await self._generate_with_retry(
            uploaded_path=uploaded_path,
            image_hash=image_hash,
            rotate_deg=rotate,
            move_forward=forward,
            vertical_tilt=tilt,
            wideangle=(lens == "wide"),
            seed=0,
            randomize_seed=True,
        )

    def optimize_image(self, image_data: bytes, max_size: int = 2048) -> bytes:
        """Optimize image for upload - resize if too large, convert to PNG."""
        try:
            img = Image.open(io.BytesIO(image_data))
            # Convert to RGB if needed
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            # Resize if too large
            w, h = img.size
            if max(w, h) > max_size:
                ratio = max_size / max(w, h)
                new_w = int(w * ratio)
                new_h = int(h * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            return buf.getvalue()
        except Exception:
            return image_data


# Singleton
hf_client = HFClient()
