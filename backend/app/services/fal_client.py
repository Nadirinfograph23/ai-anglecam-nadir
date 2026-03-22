"""fal.ai API client for Qwen Image Edit multi-angle generation.

Uses the fal-ai/qwen-image-edit-2511-multiple-angles model.
Supports full 0-360 horizontal angle and -30 to 90 vertical angle.
"""

import asyncio
import base64
import logging

import httpx

from app.config import (
    ANGLE_TIMEOUT,
    FAL_API_KEY,
    MAX_RETRIES,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
)

logger = logging.getLogger(__name__)

FAL_MODEL_ID = "fal-ai/qwen-image-edit-2511-multiple-angles"
FAL_SUBMIT_URL = f"https://queue.fal.run/{FAL_MODEL_ID}"
FAL_STATUS_URL = f"https://queue.fal.run/{FAL_MODEL_ID}/requests"


def normalize_angle_for_fal(h: float, v: float) -> tuple[float, float, float]:
    """Convert our angle system to fal.ai parameters.

    fal.ai uses:
      horizontal_angle: 0-360 (0=front, 90=right, 180=back, 270=left)
      vertical_angle: -30 to 90 (-30=low, 0=eye-level, 90=bird's-eye)
      zoom: 0-10 (0=wide, 5=medium, 10=close-up)
    """
    # Convert negative angles to 0-360 range
    horizontal = h % 360
    # Clamp vertical to fal.ai range
    vertical = max(-30.0, min(90.0, v))
    zoom = 5.0  # default medium zoom
    return horizontal, vertical, zoom


def zoom_from_lens(lens: str) -> float:
    """Convert lens type to fal.ai zoom value."""
    mapping = {"closeup": 8.0, "wide": 1.0, "normal": 5.0}
    return mapping.get(lens, 5.0)


class FalClient:
    """Client for fal.ai Qwen Image Edit multi-angle API."""

    def __init__(self) -> None:
        self._http_client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "fal.ai"

    @property
    def is_available(self) -> bool:
        return bool(FAL_API_KEY)

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
        """Generate a single angle using fal.ai API.

        Returns (image_bytes, content_type).
        """
        horizontal, vertical, zoom = normalize_angle_for_fal(h_angle, v_angle)
        zoom = zoom_from_lens(lens)

        # Encode image as base64 data URI
        b64_image = base64.b64encode(image_data).decode("utf-8")
        image_url = f"data:image/png;base64,{b64_image}"

        client = await self._get_client()
        headers = {
            "Authorization": f"Key {FAL_API_KEY}",
            "Content-Type": "application/json",
        }

        payload = {
            "image_url": image_url,
            "horizontal_angle": horizontal,
            "vertical_angle": vertical,
            "zoom": zoom,
        }

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                # Submit job to queue
                submit_resp = await client.post(
                    FAL_SUBMIT_URL,
                    json=payload,
                    headers=headers,
                )

                if submit_resp.status_code == 200:
                    # Synchronous response - result directly
                    result = submit_resp.json()
                    return await self._extract_image(result, client, headers)

                if submit_resp.status_code == 202:
                    # Queued - need to poll
                    queue_data = submit_resp.json()
                    request_id = queue_data.get("request_id")
                    if not request_id:
                        raise RuntimeError("No request_id in fal.ai queue response")
                    return await self._poll_result(request_id, client, headers)

                error_text = submit_resp.text[:300]
                logger.warning(
                    "fal.ai attempt %d failed: %d %s",
                    attempt + 1, submit_resp.status_code, error_text,
                )
                last_error = RuntimeError(f"fal.ai error: {submit_resp.status_code} - {error_text}")

            except httpx.TimeoutException:
                logger.warning("fal.ai attempt %d timed out", attempt + 1)
                last_error = RuntimeError("fal.ai request timed out")
            except Exception as e:
                logger.warning("fal.ai attempt %d error: %s", attempt + 1, str(e))
                last_error = e

            if attempt < MAX_RETRIES - 1:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                await asyncio.sleep(delay)

        raise RuntimeError(f"fal.ai failed after {MAX_RETRIES} attempts: {last_error}")

    async def _poll_result(
        self,
        request_id: str,
        client: httpx.AsyncClient,
        headers: dict[str, str],
    ) -> tuple[bytes, str]:
        """Poll for queued result."""
        status_url = f"{FAL_STATUS_URL}/{request_id}/status"
        result_url = f"{FAL_STATUS_URL}/{request_id}"

        for _ in range(60):  # max 60 polls (~2 minutes)
            await asyncio.sleep(2)

            status_resp = await client.get(status_url, headers=headers)
            if status_resp.status_code != 200:
                continue

            status_data = status_resp.json()
            status = status_data.get("status")

            if status == "COMPLETED":
                result_resp = await client.get(result_url, headers=headers)
                if result_resp.status_code == 200:
                    return await self._extract_image(result_resp.json(), client, headers)
                raise RuntimeError(f"Failed to fetch fal.ai result: {result_resp.status_code}")

            if status in ("FAILED", "CANCELLED"):
                error = status_data.get("error", "Unknown error")
                raise RuntimeError(f"fal.ai job {status}: {error}")

        raise RuntimeError("fal.ai job timed out waiting for result")

    async def _extract_image(
        self,
        result: dict,
        client: httpx.AsyncClient,
        headers: dict[str, str],
    ) -> tuple[bytes, str]:
        """Extract image bytes from fal.ai result."""
        # fal.ai returns image URL in the result
        image_info = result.get("image") or result.get("output")
        if isinstance(image_info, dict):
            image_url = image_info.get("url", "")
        elif isinstance(image_info, str):
            image_url = image_info
        else:
            # Try to find image URL in the result
            for key in ("images", "output_images"):
                val = result.get(key)
                if isinstance(val, list) and len(val) > 0:
                    item = val[0]
                    image_url = item.get("url", "") if isinstance(item, dict) else str(item)
                    break
            else:
                raise RuntimeError(f"No image found in fal.ai result: {list(result.keys())}")

        if not image_url:
            raise RuntimeError("Empty image URL from fal.ai")

        # Download the image
        img_resp = await client.get(image_url)
        if img_resp.status_code != 200:
            raise RuntimeError(f"Failed to download fal.ai image: {img_resp.status_code}")

        content_type = img_resp.headers.get("content-type", "image/webp")
        return img_resp.content, content_type


# Singleton
fal_client = FalClient()
