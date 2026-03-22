"""Stable Horde API client for image angle generation.

Uses the Stable Horde community API for img2img generation.
Free to use with anonymous API key, but slower due to community queue.
"""

import asyncio
import base64
import logging

import httpx

from app.config import (
    ANGLE_TIMEOUT,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
    STABLE_HORDE_API_KEY,
)

logger = logging.getLogger(__name__)

HORDE_API_URL = "https://stablehorde.net/api/v2"


def build_angle_prompt(h: float, v: float) -> str:
    """Build a descriptive prompt for the desired camera angle."""
    h_norm = h % 360

    if h_norm <= 22.5 or h_norm > 337.5:
        h_desc = "front view"
    elif h_norm <= 67.5:
        h_desc = "front-right view, 45 degrees rotated"
    elif h_norm <= 112.5:
        h_desc = "right side view, 90 degrees rotated"
    elif h_norm <= 157.5:
        h_desc = "back-right view, 135 degrees rotated"
    elif h_norm <= 202.5:
        h_desc = "rear back view, 180 degrees rotated"
    elif h_norm <= 247.5:
        h_desc = "back-left view, 225 degrees rotated"
    elif h_norm <= 292.5:
        h_desc = "left side view, 270 degrees rotated"
    else:
        h_desc = "front-left view, 315 degrees rotated"

    if v > 30:
        v_desc = "bird's eye view from above"
    elif v > 10:
        v_desc = "slightly elevated angle"
    elif v < -10:
        v_desc = "low angle looking up"
    else:
        v_desc = "eye level"

    return (
        f"Same object photographed from {h_desc}, {v_desc}, "
        f"photorealistic, same lighting, same background, "
        f"high quality, detailed, 8k"
    )


class StableHordeClient:
    """Client for Stable Horde community API."""

    def __init__(self) -> None:
        self._http_client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "Stable Horde"

    @property
    def is_available(self) -> bool:
        return bool(STABLE_HORDE_API_KEY)

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(ANGLE_TIMEOUT, connect=30.0),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._http_client

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def generate_angle(
        self,
        image_data: bytes,
        h_angle: float,
        v_angle: float,
        lens: str = "normal",
    ) -> tuple[bytes, str]:
        """Generate a single angle using Stable Horde img2img."""
        client = await self._get_client()
        headers = {
            "apikey": STABLE_HORDE_API_KEY,
            "Content-Type": "application/json",
        }

        b64_image = base64.b64encode(image_data).decode("utf-8")
        prompt = build_angle_prompt(h_angle, v_angle)

        # Determine denoising strength based on angle difference
        h_norm = h_angle % 360
        angle_diff = min(h_norm, 360 - h_norm)
        # More rotation = more denoising needed
        denoising = min(0.85, 0.4 + (angle_diff / 360.0))

        payload = {
            "prompt": prompt,
            "params": {
                "sampler_name": "k_euler",
                "cfg_scale": 7.5,
                "denoising_strength": denoising,
                "height": 1024,
                "width": 1024,
                "steps": 30,
                "n": 1,
            },
            "nsfw": False,
            "censor_nsfw": True,
            "source_image": b64_image,
            "source_processing": "img2img",
            "models": ["SDXL 1.0"],
            "r2": True,
        }

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    f"{HORDE_API_URL}/generate/async",
                    json=payload,
                    headers=headers,
                )

                if resp.status_code not in (200, 202):
                    error_text = resp.text[:300]
                    logger.warning(
                        "Stable Horde attempt %d failed: %d %s",
                        attempt + 1, resp.status_code, error_text,
                    )
                    last_error = RuntimeError(f"Stable Horde error: {resp.status_code}")
                    if attempt < MAX_RETRIES - 1:
                        delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                        await asyncio.sleep(delay)
                    continue

                job_data = resp.json()
                job_id = job_data.get("id")
                if not job_id:
                    raise RuntimeError("No job ID from Stable Horde")

                return await self._poll_result(job_id, client, headers)

            except httpx.TimeoutException:
                logger.warning("Stable Horde attempt %d timed out", attempt + 1)
                last_error = RuntimeError("Stable Horde request timed out")
            except Exception as e:
                logger.warning("Stable Horde attempt %d error: %s", attempt + 1, str(e))
                last_error = e

            if attempt < MAX_RETRIES - 1:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                await asyncio.sleep(delay)

        raise RuntimeError(f"Stable Horde failed after {MAX_RETRIES} attempts: {last_error}")

    async def _poll_result(
        self,
        job_id: str,
        client: httpx.AsyncClient,
        headers: dict[str, str],
    ) -> tuple[bytes, str]:
        """Poll Stable Horde for generation result."""
        for _ in range(90):  # up to ~3 minutes
            await asyncio.sleep(2)

            resp = await client.get(
                f"{HORDE_API_URL}/generate/check/{job_id}",
                headers=headers,
            )
            if resp.status_code != 200:
                continue

            data = resp.json()
            if data.get("done"):
                # Fetch the result
                result_resp = await client.get(
                    f"{HORDE_API_URL}/generate/status/{job_id}",
                    headers=headers,
                )
                if result_resp.status_code != 200:
                    raise RuntimeError(f"Failed to fetch Horde result: {result_resp.status_code}")

                result_data = result_resp.json()
                generations = result_data.get("generations", [])
                if not generations:
                    raise RuntimeError("No generations in Horde result")

                gen = generations[0]
                img_str = gen.get("img")
                if not img_str:
                    raise RuntimeError("No image in Horde generation")

                # img can be a URL (r2) or base64
                if img_str.startswith("http"):
                    img_resp = await client.get(img_str)
                    if img_resp.status_code != 200:
                        raise RuntimeError(f"Failed to download Horde image: {img_resp.status_code}")
                    content_type = img_resp.headers.get("content-type", "image/png")
                    return img_resp.content, content_type
                else:
                    img_bytes = base64.b64decode(img_str)
                    return img_bytes, "image/png"

            if data.get("faulted"):
                raise RuntimeError("Stable Horde generation faulted")

        raise RuntimeError("Stable Horde generation timed out")


# Singleton
stable_horde_client = StableHordeClient()
