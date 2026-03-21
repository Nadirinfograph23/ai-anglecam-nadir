"""HuggingFace Gradio Space client with retries, caching, rate limiting, and concurrency control."""

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
    GLOBAL_RATE_LIMIT_RPM,
    HF_API_TOKEN,
    HF_SPACE_URL,
    INTER_REQUEST_DELAY,
    MAX_CONCURRENT_GENERATIONS,
    MAX_RETRIES,
    QUOTA_RETRY_BASE_DELAY,
    QUOTA_RETRY_MAX_DELAY,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
)

logger = logging.getLogger(__name__)


class QuotaExceededError(Exception):
    """Raised when the HF API quota is exceeded."""

    def __init__(self, message: str = "API quota exceeded", retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


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


class TokenBucketRateLimiter:
    """Simple token bucket rate limiter for controlling API request rate."""

    def __init__(self, rpm: int = GLOBAL_RATE_LIMIT_RPM):
        self._tokens = float(rpm)
        self._max_tokens = float(rpm)
        self._refill_rate = rpm / 60.0  # tokens per second
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self._max_tokens, self._tokens + elapsed * self._refill_rate)
            self._last_refill = now

            if self._tokens < 1.0:
                wait_time = (1.0 - self._tokens) / self._refill_rate
                logger.info("Rate limiter: waiting %.1f seconds", wait_time)
                await asyncio.sleep(wait_time)
                self._tokens = 0.0
            else:
                self._tokens -= 1.0


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


def _is_quota_error_by_status(status_code: int) -> bool:
    """Check if an HTTP status code indicates a quota/rate-limit error."""
    return status_code in (429, 503)


def _is_quota_error_by_text(response_text: str) -> bool:
    """Check if response text contains quota/rate-limit related keywords."""
    quota_keywords = ["quota", "rate limit", "too many requests", "exceeded", "throttl"]
    lower_text = response_text.lower()
    return any(kw in lower_text for kw in quota_keywords)


def _is_quota_error(status_code: int, response_text: str) -> bool:
    """Detect if an API error is quota/rate-limit related."""
    return _is_quota_error_by_status(status_code) or _is_quota_error_by_text(response_text)


def _parse_retry_after(response_headers: httpx.Headers) -> float | None:
    """Parse Retry-After header if present."""
    retry_after = response_headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except ValueError:
            pass
    return None


class HFClient:
    """Client for interacting with the HuggingFace Gradio Space API."""

    def __init__(self) -> None:
        self._cache = ImageCache()
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
        self._http_client: httpx.AsyncClient | None = None
        self._rate_limiter = TokenBucketRateLimiter()
        self._last_request_time: float = 0.0
        self._request_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0, connect=30.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            )
        return self._http_client

    async def _throttle(self) -> None:
        """Ensure minimum delay between API requests to avoid bursting."""
        async with self._request_lock:
            now = time.monotonic()
            elapsed = now - self._last_request_time
            if elapsed < INTER_REQUEST_DELAY:
                wait_time = INTER_REQUEST_DELAY - elapsed
                await asyncio.sleep(wait_time)
            self._last_request_time = time.monotonic()

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def upload_image(self, image_data: bytes, filename: str = "input.png") -> str:
        """Upload image to HF Space and return the file path."""
        client = await self._get_client()
        token = HF_API_TOKEN

        for attempt in range(MAX_RETRIES):
            try:
                await self._rate_limiter.acquire()
                await self._throttle()

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

                if _is_quota_error(response.status_code, response.text):
                    retry_after = _parse_retry_after(response.headers)
                    raise QuotaExceededError(
                        f"Upload quota exceeded (HTTP {response.status_code})",
                        retry_after=retry_after,
                    )

                logger.warning(
                    "Upload attempt %d failed: %d %s",
                    attempt + 1, response.status_code, response.text[:200]
                )
            except QuotaExceededError as e:
                delay = e.retry_after or min(
                    QUOTA_RETRY_BASE_DELAY * (2 ** attempt), QUOTA_RETRY_MAX_DELAY
                )
                logger.warning(
                    "Upload quota exceeded on attempt %d, waiting %.1fs: %s",
                    attempt + 1, delay, str(e)
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
                else:
                    raise
            except httpx.TimeoutException:
                logger.warning("Upload attempt %d timed out", attempt + 1)
                if attempt < MAX_RETRIES - 1:
                    delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                    await asyncio.sleep(delay)
            except Exception as e:
                if isinstance(e, QuotaExceededError):
                    raise
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
        """Generate with retry logic and quota-aware backoff."""
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
            except QuotaExceededError as e:
                last_error = e
                delay = e.retry_after or min(
                    QUOTA_RETRY_BASE_DELAY * (2 ** attempt), QUOTA_RETRY_MAX_DELAY
                )
                logger.warning(
                    "Quota exceeded on attempt %d/%d for rotate=%.1f, waiting %.1fs: %s",
                    attempt + 1, MAX_RETRIES, rotate_deg, delay, str(e)
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(delay)
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

        if isinstance(last_error, QuotaExceededError):
            raise QuotaExceededError(
                "API quota exceeded after all retries. Please wait a few minutes and try again."
            )
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
        """Make the actual Gradio API call with rate limiting."""
        client = await self._get_client()
        token = HF_API_TOKEN

        # Apply rate limiting and throttling
        await self._rate_limiter.acquire()
        await self._throttle()

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
            if _is_quota_error(submit_response.status_code, submit_response.text):
                retry_after = _parse_retry_after(submit_response.headers)
                raise QuotaExceededError(
                    f"API quota exceeded (HTTP {submit_response.status_code}): "
                    f"{submit_response.text[:200]}",
                    retry_after=retry_after,
                )
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
            if _is_quota_error(result_response.status_code, result_response.text):
                retry_after = _parse_retry_after(result_response.headers)
                raise QuotaExceededError(
                    f"API quota exceeded while fetching result (HTTP {result_response.status_code})",
                    retry_after=retry_after,
                )
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
                    # Check for quota error in data
                    if isinstance(data, dict) and "error" in data:
                        error_text = str(data["error"])
                        if _is_quota_error_by_text(error_text):
                            raise QuotaExceededError(f"API quota error: {error_text}")
                    if isinstance(data, list) and len(data) > 0:
                        first = data[0]
                        if isinstance(first, dict) and "url" in first:
                            return first["url"]
                except QuotaExceededError:
                    raise
                except (json.JSONDecodeError, TypeError, KeyError):
                    continue

        if "queue" in error_msg.lower() or "quota" in error_msg.lower():
            raise QuotaExceededError(error_msg)
        raise RuntimeError(error_msg or "No result image found in SSE response")

    async def generate_all_angles(
        self,
        image_data: bytes,
        lens: str = "normal",
        on_progress: asyncio.Queue | None = None,
    ) -> list[dict]:
        """Generate all 9 angle images with controlled parallelism and rate limiting."""
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
            if isinstance(result, QuotaExceededError):
                results.append({
                    "name": angle["name"],
                    "success": False,
                    "error": "API quota exceeded. Please wait a few minutes and try again.",
                })
            elif isinstance(result, Exception):
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
