"""AngleChanger.ai API client for generating angle-changed images."""

import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

ANGLECHANGER_BASE_URL = "https://anglechanger.ai"

# Map our angle names to anglechanger.ai horizontal/vertical angle values
# anglechanger.ai uses 0-360 for horizontal (inverted) and -30..90 for vertical
ANGLE_MAPPING = {
    "Front": {"h": 0, "v": 0, "zoom": 5.0},
    "Front Right": {"h": 315, "v": 0, "zoom": 5.0},   # 360 - 45 = 315
    "Right": {"h": 270, "v": 0, "zoom": 5.0},          # 360 - 90 = 270
    "Back Right": {"h": 225, "v": 0, "zoom": 5.0},     # 360 - 135 = 225
    "Back": {"h": 180, "v": 0, "zoom": 5.0},
    "Back Left": {"h": 135, "v": 0, "zoom": 5.0},      # 360 - (-135) = 135
    "Left": {"h": 90, "v": 0, "zoom": 5.0},            # 360 - (-90) = 90
    "Front Left": {"h": 45, "v": 0, "zoom": 5.0},      # 360 - (-45) = 45
    "Top View": {"h": 0, "v": 60, "zoom": 5.0},
}


class AngleChangerClient:
    """Client for interacting with the AngleChanger.ai API."""

    def __init__(self) -> None:
        self._http_client: httpx.AsyncClient | None = None
        self._session_cookie: str | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(120.0, connect=30.0),
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
            )
        return self._http_client

    async def close(self) -> None:
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()

    async def login(self, email: str, password: str) -> bool:
        """Login to AngleChanger.ai and store session cookie."""
        client = await self._get_client()
        try:
            response = await client.post(
                f"{ANGLECHANGER_BASE_URL}/api/login.php",
                data={"email": email, "password": password},
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    # Session cookie is automatically stored in httpx client
                    self._session_cookie = str(response.cookies)
                    logger.info("Successfully logged in to AngleChanger.ai")
                    return True
                logger.warning("Login failed: %s", result.get("message", "Unknown error"))
            return False
        except Exception as e:
            logger.error("Login error: %s", str(e))
            return False

    async def upload_image(self, image_data: bytes, filename: str = "input.png") -> str:
        """Upload image to AngleChanger.ai and return the image URL."""
        client = await self._get_client()
        try:
            files = {"image": (filename, image_data, "image/png")}
            response = await client.post(
                f"{ANGLECHANGER_BASE_URL}/api/upload.php",
                files=files,
            )
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    return result["data"]["url"]
                raise RuntimeError(f"Upload failed: {result.get('message', 'Unknown error')}")
            raise RuntimeError(f"Upload HTTP error: {response.status_code}")
        except httpx.TimeoutException:
            raise RuntimeError("Upload timed out")

    async def generate_angle(
        self,
        image_url: str,
        horizontal_angle: int,
        vertical_angle: int,
        zoom: float = 5.0,
        resolution: str = "auto",
    ) -> str:
        """Generate an angle-changed image and return the result URL."""
        client = await self._get_client()

        # Step 1: Submit generation request
        payload = {
            "image_url": image_url,
            "horizontal_angle": horizontal_angle,
            "vertical_angle": vertical_angle,
            "zoom": zoom,
            "resolution": resolution,
        }

        response = await client.post(
            f"{ANGLECHANGER_BASE_URL}/api/generate.php",
            json=payload,
        )

        if response.status_code != 200:
            raise RuntimeError(f"Generate HTTP error: {response.status_code}")

        result = response.json()
        if not result.get("success"):
            raise RuntimeError(f"Generate failed: {result.get('message', 'Unknown error')}")

        request_id = result["data"]["request_id"]
        image_id = result["data"]["image_id"]

        # Step 2: Poll for completion
        return await self._poll_status(request_id, image_id)

    async def _poll_status(
        self, request_id: str, image_id: str, max_attempts: int = 60, interval: float = 2.0
    ) -> str:
        """Poll generation status until completed."""
        client = await self._get_client()

        for attempt in range(max_attempts):
            try:
                response = await client.get(
                    f"{ANGLECHANGER_BASE_URL}/api/check-status.php",
                    params={"request_id": request_id, "image_id": image_id},
                )
                if response.status_code == 200:
                    data = response.json()
                    if data.get("success") and data["data"].get("status") == "completed":
                        return data["data"]["result_url"]
                    if data.get("success") and data["data"].get("status") == "failed":
                        raise RuntimeError("Generation failed on AngleChanger.ai")
            except httpx.TimeoutException:
                logger.warning("Status poll attempt %d timed out", attempt + 1)

            await asyncio.sleep(interval)

        raise RuntimeError("Generation timed out after polling")

    async def download_image(self, url: str) -> tuple[bytes, str]:
        """Download an image from a URL. Returns (image_bytes, content_type)."""
        client = await self._get_client()
        response = await client.get(url)
        if response.status_code != 200:
            raise RuntimeError(f"Image download failed: {response.status_code}")
        content_type = response.headers.get("content-type", "image/png")
        return response.content, content_type

    async def generate_all_angles(
        self,
        image_data: bytes,
        angle_names: list[str] | None = None,
    ) -> list[dict]:
        """Generate images at all predefined angles.

        Args:
            image_data: The source image bytes.
            angle_names: Optional list of angle names to generate. If None, generates all.

        Returns:
            List of result dicts with name, success, image_data (base64), content_type, error.
        """
        import base64

        # Upload image first
        image_url = await self.upload_image(image_data)
        logger.info("Image uploaded to AngleChanger.ai: %s", image_url)

        if angle_names is None:
            angle_names = list(ANGLE_MAPPING.keys())

        results: list[dict] = []

        for angle_name in angle_names:
            mapping = ANGLE_MAPPING.get(angle_name)
            if not mapping:
                results.append({
                    "name": angle_name,
                    "success": False,
                    "error": f"Unknown angle: {angle_name}",
                })
                continue

            try:
                result_url = await self.generate_angle(
                    image_url=image_url,
                    horizontal_angle=mapping["h"],
                    vertical_angle=mapping["v"],
                    zoom=mapping["zoom"],
                )
                img_bytes, content_type = await self.download_image(result_url)
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                results.append({
                    "name": angle_name,
                    "success": True,
                    "image_data": b64,
                    "content_type": content_type,
                })
                logger.info("Generated %s via AngleChanger.ai", angle_name)

                # Use the result as input for the next generation (chain)
                image_url = result_url

            except Exception as e:
                logger.error("AngleChanger.ai generation failed for %s: %s", angle_name, str(e))
                results.append({
                    "name": angle_name,
                    "success": False,
                    "error": str(e),
                })

        return results


# Singleton
anglechanger_client = AngleChangerClient()
