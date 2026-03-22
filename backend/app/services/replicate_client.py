"""Replicate API client for image angle generation.

Uses Replicate's API to run Qwen or similar image editing models.
"""

import asyncio
import base64
import logging

import httpx

from app.config import (
    ANGLE_TIMEOUT,
    MAX_RETRIES,
    REPLICATE_API_TOKEN,
    RETRY_BASE_DELAY,
    RETRY_MAX_DELAY,
)

logger = logging.getLogger(__name__)

REPLICATE_API_URL = "https://api.replicate.com/v1/predictions"


def build_angle_prompt(h: float, v: float) -> str:
    """Build a descriptive prompt for the desired camera angle."""
    # Normalize horizontal angle
    h_norm = h % 360

    if h_norm <= 22.5 or h_norm > 337.5:
        h_desc = "front view"
    elif h_norm <= 67.5:
        h_desc = "front-right view at 45 degrees"
    elif h_norm <= 112.5:
        h_desc = "right side view at 90 degrees"
    elif h_norm <= 157.5:
        h_desc = "back-right view at 135 degrees"
    elif h_norm <= 202.5:
        h_desc = "back view at 180 degrees"
    elif h_norm <= 247.5:
        h_desc = "back-left view at 225 degrees"
    elif h_norm <= 292.5:
        h_desc = "left side view at 270 degrees"
    else:
        h_desc = "front-left view at 315 degrees"

    if v > 30:
        v_desc = "from above (bird's eye)"
    elif v > 10:
        v_desc = "slightly elevated"
    elif v < -10:
        v_desc = "from below (low angle)"
    else:
        v_desc = "at eye level"

    return f"Generate the same object/scene from a {h_desc}, {v_desc}. Keep the subject identical, only change the camera position."


class ReplicateClient:
    """Client for Replicate API."""

    def __init__(self) -> None:
        self._http_client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "Replicate"

    @property
    def is_available(self) -> bool:
        return bool(REPLICATE_API_TOKEN)

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
        """Generate a single angle using Replicate API."""
        client = await self._get_client()
        headers = {
            "Authorization": f"Bearer {REPLICATE_API_TOKEN}",
            "Content-Type": "application/json",
        }

        b64_image = base64.b64encode(image_data).decode("utf-8")
        image_uri = f"data:image/png;base64,{b64_image}"
        prompt = build_angle_prompt(h_angle, v_angle)

        # Use a general-purpose image editing model on Replicate
        payload = {
            "version": "stability-ai/sdxl:7762fd07cf82c948c1b24c680c1a42c76a4e953845da94245fbddabb7decafe1",
            "input": {
                "image": image_uri,
                "prompt": prompt,
                "guidance_scale": 7.5,
                "num_inference_steps": 30,
                "strength": 0.75,
            },
        }

        last_error: Exception | None = None

        for attempt in range(MAX_RETRIES):
            try:
                # Create prediction
                resp = await client.post(
                    REPLICATE_API_URL,
                    json=payload,
                    headers=headers,
                )

                if resp.status_code not in (200, 201):
                    error_text = resp.text[:300]
                    logger.warning(
                        "Replicate attempt %d failed: %d %s",
                        attempt + 1, resp.status_code, error_text,
                    )
                    last_error = RuntimeError(f"Replicate error: {resp.status_code}")
                    if attempt < MAX_RETRIES - 1:
                        delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                        await asyncio.sleep(delay)
                    continue

                prediction = resp.json()
                prediction_url = prediction.get("urls", {}).get("get", "")

                if not prediction_url:
                    raise RuntimeError("No prediction URL from Replicate")

                # Poll for result
                return await self._poll_result(prediction_url, client, headers)

            except httpx.TimeoutException:
                logger.warning("Replicate attempt %d timed out", attempt + 1)
                last_error = RuntimeError("Replicate request timed out")
            except Exception as e:
                logger.warning("Replicate attempt %d error: %s", attempt + 1, str(e))
                last_error = e

            if attempt < MAX_RETRIES - 1:
                delay = min(RETRY_BASE_DELAY * (2 ** attempt), RETRY_MAX_DELAY)
                await asyncio.sleep(delay)

        raise RuntimeError(f"Replicate failed after {MAX_RETRIES} attempts: {last_error}")

    async def _poll_result(
        self,
        prediction_url: str,
        client: httpx.AsyncClient,
        headers: dict[str, str],
    ) -> tuple[bytes, str]:
        """Poll Replicate for prediction result."""
        for _ in range(60):
            await asyncio.sleep(2)

            resp = await client.get(prediction_url, headers=headers)
            if resp.status_code != 200:
                continue

            data = resp.json()
            status = data.get("status")

            if status == "succeeded":
                output = data.get("output")
                if isinstance(output, list) and len(output) > 0:
                    image_url = output[0]
                elif isinstance(output, str):
                    image_url = output
                else:
                    raise RuntimeError(f"Unexpected Replicate output format: {type(output)}")

                # Download image
                img_resp = await client.get(image_url)
                if img_resp.status_code != 200:
                    raise RuntimeError(f"Failed to download Replicate image: {img_resp.status_code}")

                content_type = img_resp.headers.get("content-type", "image/png")
                return img_resp.content, content_type

            if status in ("failed", "canceled"):
                error = data.get("error", "Unknown error")
                raise RuntimeError(f"Replicate prediction {status}: {error}")

        raise RuntimeError("Replicate prediction timed out")


# Singleton
replicate_client = ReplicateClient()
