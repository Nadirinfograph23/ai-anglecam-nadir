"""Multi-provider manager with circuit breaker for image angle generation."""

import asyncio
import base64
import logging
import time
from enum import Enum

from app.config import PREDEFINED_ANGLES
from app.services.anglechanger_client import anglechanger_client
from app.services.hf_client import (
    clamp_rotate,
    compute_image_hash,
    convert_forward,
    convert_vertical,
    hf_client,
)

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    DOWN = "down"


class CircuitBreaker:
    """Circuit breaker to avoid hammering a failing provider."""

    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 60.0,
        half_open_max: int = 1,
    ) -> None:
        self._failure_count = 0
        self._failure_threshold = failure_threshold
        self._recovery_timeout = recovery_timeout
        self._half_open_max = half_open_max
        self._last_failure_time = 0.0
        self._state: ProviderStatus = ProviderStatus.HEALTHY
        self._half_open_attempts = 0

    @property
    def state(self) -> ProviderStatus:
        if self._state == ProviderStatus.DOWN:
            if time.time() - self._last_failure_time > self._recovery_timeout:
                self._state = ProviderStatus.DEGRADED
                self._half_open_attempts = 0
        return self._state

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = ProviderStatus.HEALTHY
        self._half_open_attempts = 0

    def record_failure(self) -> None:
        self._failure_count += 1
        self._last_failure_time = time.time()
        if self._failure_count >= self._failure_threshold:
            self._state = ProviderStatus.DOWN
            logger.warning(
                "Circuit breaker OPEN after %d failures", self._failure_count
            )

    def allow_request(self) -> bool:
        state = self.state
        if state == ProviderStatus.HEALTHY:
            return True
        if state == ProviderStatus.DEGRADED:
            if self._half_open_attempts < self._half_open_max:
                self._half_open_attempts += 1
                return True
            return False
        return False  # DOWN


class ProviderManager:
    """Manages multiple image generation providers with automatic fallback."""

    def __init__(self) -> None:
        self._hf_breaker = CircuitBreaker(
            failure_threshold=3,
            recovery_timeout=60.0,
        )
        self._ac_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=120.0,
        )
        self._semaphore = asyncio.Semaphore(4)  # Max concurrent generations

    async def close(self) -> None:
        await hf_client.close()
        await anglechanger_client.close()

    async def generate_angle(
        self,
        image_data: bytes,
        rotate_deg: float,
        move_forward: float,
        vertical_tilt: float,
        wideangle: bool,
        v_raw: float = 0.0,
    ) -> tuple[bytes, str, str]:
        """Generate a single angle image with automatic fallback.

        Returns:
            Tuple of (image_bytes, content_type, provider_name)
        """
        async with self._semaphore:
            # Try HuggingFace first if circuit breaker allows
            if self._hf_breaker.allow_request():
                try:
                    result = await self._generate_via_hf(
                        image_data, rotate_deg, move_forward,
                        vertical_tilt, wideangle,
                    )
                    self._hf_breaker.record_success()
                    return result[0], result[1], "huggingface"
                except Exception as e:
                    logger.warning("HF generation failed: %s", str(e))
                    self._hf_breaker.record_failure()

            # Fallback to AngleChanger.ai
            if self._ac_breaker.allow_request():
                try:
                    result = await self._generate_via_anglechanger(
                        image_data, rotate_deg, v_raw,
                    )
                    self._ac_breaker.record_success()
                    return result[0], result[1], "anglechanger"
                except Exception as e:
                    logger.warning("AngleChanger generation failed: %s", str(e))
                    self._ac_breaker.record_failure()

            # Both providers failed - try HF one more time as last resort
            try:
                result = await self._generate_via_hf(
                    image_data, rotate_deg, move_forward,
                    vertical_tilt, wideangle,
                )
                self._hf_breaker.record_success()
                return result[0], result[1], "huggingface"
            except Exception as e:
                raise RuntimeError(
                    f"All providers failed. Last error: {str(e)}"
                ) from e

    async def _generate_via_hf(
        self,
        image_data: bytes,
        rotate_deg: float,
        move_forward: float,
        vertical_tilt: float,
        wideangle: bool,
    ) -> tuple[bytes, str]:
        """Generate via HuggingFace Gradio Space."""
        image_hash = compute_image_hash(image_data)
        optimized = hf_client.optimize_image(image_data)
        uploaded_path = await hf_client.upload_image(optimized)

        return await hf_client.generate_angle(
            uploaded_path=uploaded_path,
            image_hash=image_hash,
            rotate_deg=rotate_deg,
            move_forward=move_forward,
            vertical_tilt=vertical_tilt,
            wideangle=wideangle,
        )

    async def _generate_via_anglechanger(
        self,
        image_data: bytes,
        rotate_deg: float,
        v_raw: float,
    ) -> tuple[bytes, str]:
        """Generate via anglechanger.ai as fallback."""
        optimized = hf_client.optimize_image(image_data)
        image_url = await anglechanger_client.upload_image(optimized)

        # Convert angles for anglechanger.ai format
        h_ac, v_ac = anglechanger_client.convert_angle_for_ac(rotate_deg, v_raw)

        return await anglechanger_client.generate_angle(
            image_url=image_url,
            horizontal_angle=h_ac,
            vertical_angle=v_ac,
            zoom=5.0,
        )

    async def generate_all_angles(
        self,
        image_data: bytes,
        lens: str = "normal",
    ) -> list[dict]:
        """Generate all predefined angles with automatic fallback per angle."""
        forward = convert_forward(lens)
        is_wide = lens == "wide"

        tasks = []
        for angle in PREDEFINED_ANGLES:
            rotate = clamp_rotate(float(angle["h"]))
            tilt = convert_vertical(float(angle["v"]))

            task = self._generate_single_angle_task(
                image_data=image_data,
                angle_name=angle["name"],
                rotate_deg=rotate,
                move_forward=forward,
                vertical_tilt=tilt,
                wideangle=is_wide,
                v_raw=float(angle["v"]),
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        output = []
        for i, result in enumerate(results):
            angle = PREDEFINED_ANGLES[i]
            if isinstance(result, Exception):
                output.append({
                    "name": angle["name"],
                    "success": False,
                    "error": str(result),
                })
            else:
                img_bytes, content_type, provider = result
                b64 = base64.b64encode(img_bytes).decode("utf-8")
                output.append({
                    "name": angle["name"],
                    "success": True,
                    "image_data": b64,
                    "content_type": content_type,
                    "provider": provider,
                })

        return output

    async def _generate_single_angle_task(
        self,
        image_data: bytes,
        angle_name: str,
        rotate_deg: float,
        move_forward: float,
        vertical_tilt: float,
        wideangle: bool,
        v_raw: float,
    ) -> tuple[bytes, str, str]:
        """Generate a single angle with fallback. Used in generate_all_angles."""
        try:
            return await self.generate_angle(
                image_data=image_data,
                rotate_deg=rotate_deg,
                move_forward=move_forward,
                vertical_tilt=vertical_tilt,
                wideangle=wideangle,
                v_raw=v_raw,
            )
        except Exception as e:
            logger.error("All providers failed for %s: %s", angle_name, str(e))
            raise

    def get_provider_status(self) -> dict:
        """Return current health status of all providers."""
        return {
            "huggingface": self._hf_breaker.state.value,
            "anglechanger": self._ac_breaker.state.value,
        }


# Singleton
provider_manager = ProviderManager()
