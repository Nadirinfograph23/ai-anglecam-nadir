"""AngleChanger.ai client - fallback provider for image angle generation."""

import asyncio
import logging
import time

import httpx

logger = logging.getLogger(__name__)

ANGLECHANGER_BASE_URL = "https://anglechanger.ai"
UPLOAD_ENDPOINT = f"{ANGLECHANGER_BASE_URL}/api/upload.php"
GENERATE_ENDPOINT = f"{ANGLECHANGER_BASE_URL}/api/generate.php"
CHECK_STATUS_ENDPOINT = f"{ANGLECHANGER_BASE_URL}/api/check-status.php"

# Polling configuration
MAX_POLL_ATTEMPTS = 60
POLL_INTERVAL = 2.0  # seconds between status checks


class AngleChangerClient:
    """Client for interacting with anglechanger.ai as a fallback provider."""

    def __init__(self) -> None:
        self._http_client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=30.0),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "application/json, text/plain, */*",
                    "Origin": ANGLECHANGER_BASE_URL,
                    "Referer": f"{ANGLECHANGER_BASE_URL}/",
                },
            )
        return self._http_client

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def upload_image(self, image_data: bytes, filename: str = "input.png") -> str:
        """Upload image to anglechanger.ai and return the image URL."""
        client = await self._get_client()

        files = {"image": (filename, image_data, "image/png")}
        response = await client.post(UPLOAD_ENDPOINT, files=files)

        if response.status_code != 200:
            raise RuntimeError(
                f"AngleChanger upload failed: {response.status_code} - {response.text[:200]}"
            )

        result = response.json()
        if not result.get("success"):
            raise RuntimeError(
                f"AngleChanger upload error: {result.get('message', 'Unknown error')}"
            )

        image_url = result.get("data", {}).get("url")
        if not image_url:
            raise RuntimeError("No image URL in AngleChanger upload response")

        logger.info("AngleChanger upload successful: %s", image_url)
        return image_url

    async def generate_angle(
        self,
        image_url: str,
        horizontal_angle: float,
        vertical_angle: float,
        zoom: float = 5.0,
        resolution: str = "auto",
    ) -> tuple[bytes, str]:
        """Generate a single angle image via anglechanger.ai.

        Args:
            image_url: URL of the uploaded image on anglechanger.ai
            horizontal_angle: Horizontal rotation angle (0-360)
            vertical_angle: Vertical tilt angle (-30 to 90)
            zoom: Zoom level (default 5.0)
            resolution: Resolution setting ('sd', 'hd', 'fullhd', '2k', 'auto')

        Returns:
            Tuple of (image_bytes, content_type)
        """
        client = await self._get_client()

        # anglechanger.ai inverts horizontal angle for its API
        inverted_h = (360 - int(horizontal_angle)) % 360

        # Clamp vertical angle to anglechanger.ai's range
        clamped_v = max(-30, min(90, int(vertical_angle)))

        payload = {
            "image_url": image_url,
            "horizontal_angle": inverted_h,
            "vertical_angle": clamped_v,
            "zoom": zoom,
            "resolution": resolution,
        }

        logger.info(
            "AngleChanger generate: h=%d, v=%d, zoom=%.1f",
            inverted_h, clamped_v, zoom,
        )

        response = await client.post(
            GENERATE_ENDPOINT,
            json=payload,
            headers={"Content-Type": "application/json"},
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"AngleChanger generate failed: {response.status_code} - {response.text[:200]}"
            )

        result = response.json()
        if not result.get("success"):
            error_msg = result.get("message") or result.get("error") or "Generation failed"
            raise RuntimeError(f"AngleChanger generate error: {error_msg}")

        request_id = result.get("data", {}).get("request_id")
        image_id = result.get("data", {}).get("image_id")

        if not request_id or not image_id:
            raise RuntimeError("No request_id/image_id in AngleChanger generate response")

        # Poll for completion
        result_url = await self._poll_status(request_id, image_id)

        # Download the result image
        image_bytes, content_type = await self._download_image(result_url)
        return image_bytes, content_type

    async def _poll_status(self, request_id: str, image_id: str) -> str:
        """Poll anglechanger.ai for generation completion."""
        client = await self._get_client()

        for attempt in range(MAX_POLL_ATTEMPTS):
            response = await client.get(
                CHECK_STATUS_ENDPOINT,
                params={"request_id": request_id, "image_id": image_id},
            )

            if response.status_code != 200:
                logger.warning(
                    "AngleChanger status check failed: %d", response.status_code
                )
                await asyncio.sleep(POLL_INTERVAL)
                continue

            data = response.json()

            if data.get("success") and data.get("data", {}).get("status") == "completed":
                result_url = data["data"].get("result_url")
                if result_url:
                    logger.info("AngleChanger generation completed: %s", result_url)
                    return result_url
                raise RuntimeError("Completed but no result_url")

            if data.get("data", {}).get("status") == "failed":
                raise RuntimeError(
                    f"AngleChanger generation failed: {data.get('data', {}).get('error', 'Unknown')}"
                )

            # Still processing
            await asyncio.sleep(POLL_INTERVAL)

        raise RuntimeError(
            f"AngleChanger generation timed out after {MAX_POLL_ATTEMPTS * POLL_INTERVAL}s"
        )

    async def _download_image(self, url: str) -> tuple[bytes, str]:
        """Download the generated image from anglechanger.ai."""
        client = await self._get_client()

        # Make URL absolute if relative
        if not url.startswith("http"):
            url = f"{ANGLECHANGER_BASE_URL}/{url.lstrip('/')}"

        response = await client.get(url)
        if response.status_code != 200:
            raise RuntimeError(
                f"AngleChanger image download failed: {response.status_code}"
            )

        content_type = response.headers.get("content-type", "image/png")
        return response.content, content_type

    def convert_angle_for_ac(self, h_deg: float, v_deg: float) -> tuple[float, float]:
        """Convert internal angle format to anglechanger.ai format.

        Internal format: h in [-180, 180], v as tilt factor [-1, 1]
        AngleChanger format: h in [0, 360], v in [-30, 90]
        """
        # Convert horizontal: [-180, 180] -> [0, 360]
        h_ac = h_deg % 360
        if h_ac < 0:
            h_ac += 360

        # Convert vertical: tilt factor [-1, 1] -> degrees [-30, 90]
        # Our v is already in the 0-60 range from config, convert_vertical maps to [-1, 1]
        # For anglechanger.ai we need the raw degree value
        v_ac = max(-30, min(90, v_deg))

        return h_ac, v_ac


# Singleton
anglechanger_client = AngleChangerClient()
