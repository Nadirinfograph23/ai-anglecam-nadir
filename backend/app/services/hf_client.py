"""HuggingFace Gradio Space client with retries, caching, multi-token fallback, and queue."""

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
    GITHUB_RAW_BRANCH,
    GITHUB_RAW_REPO,
    GITHUB_RAW_TOKEN,
    HF_API_TOKENS,
    HF_SPACE_URL,
    MAX_CONCURRENT_GENERATIONS,
    MAX_QUEUE_SIZE,
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

    @property
    def stats(self) -> dict:
        """Return cache statistics."""
        now = time.time()
        valid = sum(1 for e in self._cache.values() if now - e.created_at <= self._ttl)
        return {"size": len(self._cache), "valid": valid, "max_size": self._max_size}


class TokenRotator:
    """Rotate through multiple HF API tokens, skipping failed ones temporarily."""

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens if tokens else [""]
        self._current_index = 0
        self._cooldowns: dict[int, float] = {}
        self._lock = asyncio.Lock()

    @property
    def current_token(self) -> str:
        return self._tokens[self._current_index] if self._tokens else ""

    async def get_token(self) -> str:
        """Get the next available token, skipping cooled-down ones."""
        async with self._lock:
            now = time.time()
            for i in range(len(self._tokens)):
                idx = (self._current_index + i) % len(self._tokens)
                cooldown_until = self._cooldowns.get(idx, 0)
                if now >= cooldown_until:
                    self._current_index = idx
                    return self._tokens[idx]
            # All tokens on cooldown, use the one with shortest remaining cooldown
            min_idx = min(self._cooldowns, key=self._cooldowns.get, default=0)
            self._current_index = min_idx
            return self._tokens[min_idx]

    async def mark_failed(self, token: str, cooldown_seconds: float = 60.0) -> None:
        """Mark a token as failed, put it on cooldown."""
        async with self._lock:
            for i, t in enumerate(self._tokens):
                if t == token:
                    self._cooldowns[i] = time.time() + cooldown_seconds
                    logger.warning("Token %d put on %.0fs cooldown", i, cooldown_seconds)
                    break

    async def mark_success(self, token: str) -> None:
        """Clear cooldown for a successful token."""
        async with self._lock:
            for i, t in enumerate(self._tokens):
                if t == token:
                    self._cooldowns.pop(i, None)
                    break

    @property
    def token_count(self) -> int:
        return len(self._tokens)


class JobQueue:
    """Simple async job queue with concurrency control."""

    def __init__(self, max_concurrent: int = MAX_CONCURRENT_GENERATIONS, max_size: int = MAX_QUEUE_SIZE):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._queue_size = 0
        self._max_size = max_size
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Try to acquire a slot. Returns False if queue is full."""
        async with self._lock:
            if self._queue_size >= self._max_size:
                return False
            self._queue_size += 1

        await self._semaphore.acquire()
        return True

    def release(self) -> None:
        """Release a slot."""
        self._semaphore.release()
        self._queue_size = max(0, self._queue_size - 1)

    @property
    def pending(self) -> int:
        return self._queue_size

    @property
    def max_size(self) -> int:
        return self._max_size


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
        self._token_rotator = TokenRotator(HF_API_TOKENS)
        self._job_queue = JobQueue()
        self._http_client: httpx.AsyncClient | None = None

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

    def get_status(self) -> dict:
        """Return current system status."""
        return {
            "cache": self._cache.stats,
            "queue_pending": self._job_queue.pending,
            "queue_max": self._job_queue.max_size,
            "token_count": self._token_rotator.token_count,
        }

    async def upload_image(self, image_data: bytes, filename: str = "input.png") -> str:
        """Upload image to HF Space and return the file path."""
        client = await self._get_client()

        for attempt in range(MAX_RETRIES):
            token = await self._token_rotator.get_token()
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
                        await self._token_rotator.mark_success(token)
                        return result[0]
                    raise ValueError(f"Unexpected upload response: {result}")

                if response.status_code in (429, 503):
                    await self._token_rotator.mark_failed(token, cooldown_seconds=120.0)
                    logger.warning("Upload rate-limited (token rotated), attempt %d", attempt + 1)
                else:
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

        # Acquire queue slot with concurrency control
        acquired = await self._job_queue.acquire()
        if not acquired:
            raise RuntimeError("Server busy - too many pending requests. Please try again shortly.")

        try:
            return await self._generate_with_retry(
                uploaded_path, image_hash, rotate_deg, move_forward,
                vertical_tilt, wideangle, seed, randomize_seed
            )
        finally:
            self._job_queue.release()

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
        """Generate with retry logic and token rotation."""
        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            token = await self._token_rotator.get_token()
            try:
                result = await self._call_gradio_api(
                    uploaded_path, rotate_deg, move_forward,
                    vertical_tilt, wideangle, seed, randomize_seed,
                    token=token,
                )
                # Cache the result
                self._cache.put(
                    image_hash, rotate_deg, move_forward, vertical_tilt, wideangle,
                    result[0], result[1]
                )
                await self._token_rotator.mark_success(token)

                # Save to GitHub RAW in background (fire and forget)
                if GITHUB_RAW_REPO and GITHUB_RAW_TOKEN:
                    asyncio.create_task(
                        self._save_to_github(
                            image_hash, rotate_deg, move_forward,
                            vertical_tilt, wideangle, result[0], result[1]
                        )
                    )

                return result
            except Exception as e:
                last_error = e
                error_str = str(e)
                logger.warning(
                    "Generation attempt %d/%d failed for rotate=%.1f: %s",
                    attempt + 1, MAX_RETRIES, rotate_deg, error_str
                )
                # Rotate token on rate limit or auth errors
                if any(code in error_str for code in ["429", "503", "401", "403"]):
                    await self._token_rotator.mark_failed(token, cooldown_seconds=120.0)
                    logger.info("Token rotated due to rate limit/auth error")

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
        token: str = "",
    ) -> tuple[bytes, str]:
        """Make the actual Gradio API call."""
        client = await self._get_client()

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

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        submit_response = await client.post(
            f"{HF_SPACE_URL}/gradio_api/call/maybe_infer",
            json=payload,
            headers=headers,
        )

        if submit_response.status_code == 429:
            raise RuntimeError("429 Rate limited by HF Space")
        if submit_response.status_code == 503:
            raise RuntimeError("503 HF Space is temporarily unavailable")
        if submit_response.status_code != 200:
            raise RuntimeError(
                f"Submit failed: {submit_response.status_code} - {submit_response.text[:300]}"
            )

        event_id = submit_response.json().get("event_id")
        if not event_id:
            raise RuntimeError("No event_id in submit response")

        # Step 2: Poll for result (SSE stream)
        result_headers: dict[str, str] = {}
        if token:
            result_headers["Authorization"] = f"Bearer {token}"

        result_response = await client.get(
            f"{HF_SPACE_URL}/gradio_api/call/maybe_infer/{event_id}",
            headers=result_headers,
        )

        if result_response.status_code != 200:
            raise RuntimeError(
                f"Result fetch failed: {result_response.status_code}"
            )

        # Parse SSE response
        image_url = self._parse_sse_response(result_response.text)

        # Step 3: Download the generated image
        download_headers: dict[str, str] = {}
        if token:
            download_headers["Authorization"] = f"Bearer {token}"

        image_response = await client.get(
            image_url,
            headers=download_headers,
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
        error_data = ""

        for i, line in enumerate(lines):
            if line.startswith("event: error"):
                error_msg = "API returned an error"
                # Try to get error details from the next data line
                if i + 1 < len(lines) and lines[i + 1].startswith("data: "):
                    error_data = lines[i + 1][6:].strip()
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

        if error_data:
            raise RuntimeError(f"HF API error: {error_data[:200]}")
        raise RuntimeError(error_msg or "No result image found in SSE response")

    async def _save_to_github(
        self,
        image_hash: str,
        rotate: float,
        forward: float,
        tilt: float,
        wide: bool,
        data: bytes,
        content_type: str,
    ) -> None:
        """Save generated image to GitHub repo for RAW URL caching."""
        try:
            ext = "webp" if "webp" in content_type else "png"
            filename = f"{image_hash}_{rotate}_{forward}_{tilt}_{wide}.{ext}"
            path = f"generated/{filename}"

            client = await self._get_client()
            b64_content = base64.b64encode(data).decode("utf-8")

            response = await client.put(
                f"https://api.github.com/repos/{GITHUB_RAW_REPO}/contents/{path}",
                json={
                    "message": f"Auto-save generated image {filename}",
                    "content": b64_content,
                    "branch": GITHUB_RAW_BRANCH,
                },
                headers={
                    "Authorization": f"token {GITHUB_RAW_TOKEN}",
                    "Accept": "application/vnd.github.v3+json",
                },
                timeout=30.0,
            )
            if response.status_code in (200, 201):
                logger.info("Saved to GitHub: %s", path)
            else:
                logger.debug("GitHub save skipped: %d", response.status_code)
        except Exception as e:
            logger.debug("GitHub save failed (non-critical): %s", str(e))

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

        # Run with controlled parallelism (queue in generate_angle)
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
