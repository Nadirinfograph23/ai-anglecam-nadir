"""HuggingFace Gradio Space client with multi-token rotation, fallback spaces,
retries, caching, and concurrency control for massive quota boost."""

import asyncio
import base64
import hashlib
import io
import json
import logging
import random
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
    FALLBACK_SPACE_URLS,
    HF_API_TOKENS,
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
    """Normalize rotation to [-180, 180] range without clamping to [-90, 90].

    The Gradio API supports full rotation values. Previous clamping to [-90, 90]
    caused Back, Back Right, and Back Left to produce duplicate images.
    """
    # Normalize to [-180, 180]
    while deg > 180:
        deg -= 360
    while deg < -180:
        deg += 360
    return deg


def convert_vertical(v: float) -> float:
    return max(-1.0, min(1.0, v / 60.0))


def convert_forward(lens: str) -> float:
    mapping = {"closeup": 5.0, "wide": 0.0, "normal": 2.0}
    return mapping.get(lens, 2.0)


class TokenRotator:
    """Round-robin token rotation with rate-limit tracking per token.

    Distributes API calls across multiple HF tokens to massively increase
    aggregate quota. Tokens that hit rate limits are temporarily excluded.
    """

    def __init__(self, tokens: list[str]) -> None:
        self._tokens = tokens if tokens else [""]
        self._index = 0
        self._lock = asyncio.Lock()
        # Track rate-limited tokens: token -> cooldown_until timestamp
        self._cooldowns: dict[str, float] = {}
        self._cooldown_duration = 60.0  # seconds to wait after rate limit

    async def get_token(self) -> str:
        """Get the next available token using round-robin rotation."""
        async with self._lock:
            now = time.time()
            # Try to find a non-rate-limited token
            for _ in range(len(self._tokens)):
                token = self._tokens[self._index]
                self._index = (self._index + 1) % len(self._tokens)
                cooldown_until = self._cooldowns.get(token, 0)
                if now >= cooldown_until:
                    return token

            # All tokens are rate-limited, return the one with earliest cooldown
            earliest_token = min(self._tokens, key=lambda t: self._cooldowns.get(t, 0))
            wait_time = self._cooldowns.get(earliest_token, 0) - now
            if wait_time > 0:
                logger.info("All tokens rate-limited, waiting %.1fs", wait_time)
            return earliest_token

    def mark_rate_limited(self, token: str) -> None:
        """Mark a token as rate-limited for the cooldown period."""
        self._cooldowns[token] = time.time() + self._cooldown_duration
        logger.warning("Token %s...%s rate-limited, cooling down for %.0fs",
                        token[:8], token[-4:] if len(token) > 8 else "", self._cooldown_duration)

    def clear_cooldown(self, token: str) -> None:
        """Clear rate-limit cooldown for a token after successful use."""
        self._cooldowns.pop(token, None)


class SpaceRotator:
    """Rotate through multiple Gradio Space URLs for failover.

    When the primary space is overloaded or rate-limited, automatically
    falls back to alternative compatible spaces.
    """

    def __init__(self, primary_url: str, fallback_urls: list[str]) -> None:
        self._spaces = [primary_url] + [u for u in fallback_urls if u != primary_url]
        self._failed: dict[str, float] = {}
        self._failure_cooldown = 120.0  # seconds before retrying a failed space

    def get_spaces(self) -> list[str]:
        """Return list of available spaces, prioritizing non-failed ones."""
        now = time.time()
        available = []
        failed_but_recoverable = []

        for space in self._spaces:
            fail_until = self._failed.get(space, 0)
            if now >= fail_until:
                available.append(space)
            else:
                failed_but_recoverable.append(space)

        # Return available first, then failed ones as last resort
        return available + failed_but_recoverable

    def mark_failed(self, space_url: str) -> None:
        """Mark a space as temporarily failed."""
        self._failed[space_url] = time.time() + self._failure_cooldown
        logger.warning("Space %s marked as failed, cooldown %.0fs", space_url, self._failure_cooldown)

    def mark_success(self, space_url: str) -> None:
        """Clear failure status for a space."""
        self._failed.pop(space_url, None)


class HFClient:
    """Client for interacting with HuggingFace Gradio Space API.

    Features:
    - Multi-token rotation for massive quota increase
    - Fallback to alternative Gradio spaces
    - Per-request token + space selection
    - Automatic retry with exponential backoff
    - LRU cache with TTL
    - Concurrency control via semaphore
    """

    def __init__(self) -> None:
        self._cache = ImageCache()
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
        self._http_client: httpx.AsyncClient | None = None
        self._token_rotator = TokenRotator(HF_API_TOKENS)
        self._space_rotator = SpaceRotator(HF_SPACE_URL, FALLBACK_SPACE_URLS)

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

    async def upload_image(self, image_data: bytes, filename: str = "input.png",
                           space_url: str | None = None, token: str | None = None) -> str:
        """Upload image to HF Space and return the file path."""
        client = await self._get_client()
        target_space = space_url or HF_SPACE_URL
        auth_token = token or await self._token_rotator.get_token()

        for attempt in range(MAX_RETRIES):
            try:
                files = {"files": (filename, image_data, "image/png")}
                headers: dict[str, str] = {}
                if auth_token:
                    headers["Authorization"] = f"Bearer {auth_token}"

                response = await client.post(
                    f"{target_space}/gradio_api/upload",
                    files=files,
                    headers=headers,
                )
                if response.status_code == 200:
                    result = response.json()
                    if isinstance(result, list) and len(result) > 0:
                        return result[0]
                    raise ValueError(f"Unexpected upload response: {result}")

                if response.status_code == 429:
                    if auth_token:
                        self._token_rotator.mark_rate_limited(auth_token)
                    auth_token = await self._token_rotator.get_token()

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
        """Generate with retry logic across multiple tokens and spaces."""
        last_error: Exception | None = None
        spaces = self._space_rotator.get_spaces()
        total_attempts = MAX_RETRIES * max(1, len(spaces))

        for attempt in range(min(total_attempts, MAX_RETRIES * 3)):
            # Get next token from rotation
            token = await self._token_rotator.get_token()
            # Rotate through available spaces
            space_url = spaces[attempt % len(spaces)] if spaces else HF_SPACE_URL

            try:
                result = await self._call_gradio_api(
                    uploaded_path, rotate_deg, move_forward,
                    vertical_tilt, wideangle, seed, randomize_seed,
                    space_url=space_url, token=token,
                )
                # Success - clear any cooldowns
                self._token_rotator.clear_cooldown(token)
                self._space_rotator.mark_success(space_url)
                # Cache the result
                self._cache.put(
                    image_hash, rotate_deg, move_forward, vertical_tilt, wideangle,
                    result[0], result[1]
                )
                return result
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Handle rate limiting
                if "429" in error_str or "quota" in error_str or "rate" in error_str:
                    self._token_rotator.mark_rate_limited(token)
                    logger.warning(
                        "Rate limited on attempt %d/%d (space=%s): %s",
                        attempt + 1, total_attempts, space_url, str(e)
                    )
                elif "500" in error_str or "503" in error_str or "502" in error_str:
                    self._space_rotator.mark_failed(space_url)
                    logger.warning(
                        "Space error on attempt %d/%d (space=%s): %s",
                        attempt + 1, total_attempts, space_url, str(e)
                    )
                else:
                    logger.warning(
                        "Generation attempt %d/%d failed for rotate=%.1f: %s",
                        attempt + 1, total_attempts, rotate_deg, str(e)
                    )

                if attempt < total_attempts - 1:
                    # Shorter delay when rotating tokens/spaces
                    delay = min(RETRY_BASE_DELAY * (1.5 ** (attempt // max(1, len(spaces)))), RETRY_MAX_DELAY)
                    # Add jitter to prevent thundering herd
                    delay += random.uniform(0, 1.0)
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)

        raise RuntimeError(
            f"Generation failed after {total_attempts} attempts across "
            f"{len(spaces)} spaces: {last_error}"
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
        space_url: str | None = None,
        token: str | None = None,
    ) -> tuple[bytes, str]:
        """Make the actual Gradio API call to a specific space with a specific token."""
        client = await self._get_client()
        target_space = space_url or HF_SPACE_URL
        auth_token = token or ""

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
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        submit_response = await client.post(
            f"{target_space}/gradio_api/call/maybe_infer",
            json=payload,
            headers=headers,
        )

        if submit_response.status_code == 429:
            raise RuntimeError(f"429 Rate limited on {target_space}")

        if submit_response.status_code != 200:
            raise RuntimeError(
                f"Submit failed: {submit_response.status_code} - {submit_response.text[:300]}"
            )

        event_id = submit_response.json().get("event_id")
        if not event_id:
            raise RuntimeError("No event_id in submit response")

        # Step 2: Poll for result (SSE stream)
        poll_headers: dict[str, str] = {}
        if auth_token:
            poll_headers["Authorization"] = f"Bearer {auth_token}"

        result_response = await client.get(
            f"{target_space}/gradio_api/call/maybe_infer/{event_id}",
            headers=poll_headers,
        )

        if result_response.status_code == 429:
            raise RuntimeError(f"429 Rate limited polling {target_space}")

        if result_response.status_code != 200:
            raise RuntimeError(
                f"Result fetch failed: {result_response.status_code}"
            )

        # Parse SSE response
        image_url = self._parse_sse_response(result_response.text)

        # Step 3: Download the generated image
        dl_headers: dict[str, str] = {}
        if auth_token:
            dl_headers["Authorization"] = f"Bearer {auth_token}"

        image_response = await client.get(
            image_url,
            headers=dl_headers,
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
                    # Check for error messages in data
                    if isinstance(data, dict) and "error" in data:
                        error_msg = str(data["error"])
                        continue
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
            # Use per-angle forward override if defined, otherwise use lens default
            angle_forward = angle.get("forward") if angle.get("forward") is not None else forward

            task = self._generate_angle_task(
                uploaded_path=uploaded_path,
                image_hash=image_hash,
                angle_name=angle["name"],
                rotate_deg=rotate,
                move_forward=angle_forward,
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
