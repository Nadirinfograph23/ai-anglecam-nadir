"""Multi-provider manager with automatic fallback.

Manages multiple image generation providers and automatically falls back
to the next provider when one fails. Supports:
- fal.ai (primary, fastest, full 360° rotation)
- HuggingFace Space (Gradio API)
- Replicate
- Stable Horde (community, free)
"""

import asyncio
import base64
import hashlib
import io
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from PIL import Image

from app.config import (
    ANGLE_TIMEOUT,
    CACHE_MAX_SIZE,
    CACHE_TTL_SECONDS,
    MAX_CONCURRENT_GENERATIONS,
    PREDEFINED_ANGLES,
    PROVIDER_ORDER,
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

    def _make_key(self, image_hash: str, h: float, v: float, lens: str) -> str:
        return f"{image_hash}:{h}:{v}:{lens}"

    def get(self, image_hash: str, h: float, v: float, lens: str) -> CacheEntry | None:
        key = self._make_key(image_hash, h, v, lens)
        entry = self._cache.get(key)
        if entry is None:
            return None
        if time.time() - entry.created_at > self._ttl:
            del self._cache[key]
            return None
        self._cache.move_to_end(key)
        return entry

    def put(self, image_hash: str, h: float, v: float, lens: str,
            data: bytes, content_type: str) -> None:
        key = self._make_key(image_hash, h, v, lens)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = CacheEntry(data=data, content_type=content_type)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)


def compute_image_hash(image_data: bytes) -> str:
    return hashlib.sha256(image_data).hexdigest()[:16]


def optimize_image(image_data: bytes, max_size: int = 2048) -> bytes:
    """Optimize image for upload - resize if too large, convert to PNG."""
    try:
        img = Image.open(io.BytesIO(image_data))
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
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


class ProviderManager:
    """Manages multiple providers with automatic fallback."""

    def __init__(self) -> None:
        self._cache = ImageCache()
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)
        self._providers: list[object] = []
        self._initialized = False

    def _init_providers(self) -> None:
        """Lazily initialize providers based on config."""
        if self._initialized:
            return
        self._initialized = True

        from app.services.fal_client import fal_client
        from app.services.hf_client import hf_client
        from app.services.replicate_client import replicate_client
        from app.services.stable_horde_client import stable_horde_client

        provider_map = {
            "fal": fal_client,
            "hf": hf_client,
            "replicate": replicate_client,
            "stablehorde": stable_horde_client,
        }

        order = [p.strip() for p in PROVIDER_ORDER.split(",") if p.strip()]
        for name in order:
            provider = provider_map.get(name)
            if provider and provider.is_available:
                self._providers.append(provider)
                logger.info("Provider enabled: %s (%s)", name, provider.name)
            else:
                logger.info("Provider skipped (not available): %s", name)

        if not self._providers:
            # Always have HF as last resort (even without token, it may work for public spaces)
            self._providers.append(hf_client)
            logger.warning("No providers configured with API keys, using HF Space as fallback")

    async def close(self) -> None:
        for provider in self._providers:
            if hasattr(provider, "close"):
                await provider.close()

    async def generate_angle(
        self,
        image_data: bytes,
        image_hash: str,
        h_angle: float,
        v_angle: float,
        lens: str = "normal",
    ) -> tuple[bytes, str]:
        """Generate a single angle with provider fallback.

        Tries each provider in order until one succeeds.
        Returns (image_bytes, content_type).
        """
        self._init_providers()

        # Check cache first
        cached = self._cache.get(image_hash, h_angle, v_angle, lens)
        if cached is not None:
            logger.info("Cache hit for hash=%s h=%.1f v=%.1f lens=%s",
                        image_hash, h_angle, v_angle, lens)
            return cached.data, cached.content_type

        async with self._semaphore:
            return await self._generate_with_fallback(
                image_data, image_hash, h_angle, v_angle, lens
            )

    async def _generate_with_fallback(
        self,
        image_data: bytes,
        image_hash: str,
        h_angle: float,
        v_angle: float,
        lens: str,
    ) -> tuple[bytes, str]:
        """Try each provider in order with timeout."""
        errors: list[str] = []

        for provider in self._providers:
            try:
                logger.info(
                    "Trying provider %s for h=%.1f v=%.1f lens=%s",
                    provider.name, h_angle, v_angle, lens,
                )

                # Apply timeout per provider attempt
                result = await asyncio.wait_for(
                    self._call_provider(provider, image_data, h_angle, v_angle, lens),
                    timeout=ANGLE_TIMEOUT,
                )

                # Cache successful result
                self._cache.put(
                    image_hash, h_angle, v_angle, lens,
                    result[0], result[1],
                )

                logger.info("Provider %s succeeded for h=%.1f v=%.1f",
                            provider.name, h_angle, v_angle)
                return result

            except asyncio.TimeoutError:
                msg = f"{provider.name}: timed out after {ANGLE_TIMEOUT}s"
                logger.warning(msg)
                errors.append(msg)
            except Exception as e:
                msg = f"{provider.name}: {str(e)}"
                logger.warning("Provider %s failed: %s", provider.name, str(e))
                errors.append(msg)

        raise RuntimeError(
            f"All providers failed for h={h_angle} v={v_angle}: "
            + "; ".join(errors)
        )

    async def _call_provider(
        self,
        provider: object,
        image_data: bytes,
        h_angle: float,
        v_angle: float,
        lens: str,
    ) -> tuple[bytes, str]:
        """Call a specific provider to generate an angle."""
        # HF client has a different interface - needs uploaded_path
        from app.services.hf_client import HFClient
        if isinstance(provider, HFClient):
            return await provider.generate_angle_direct(
                image_data=image_data,
                h_angle=h_angle,
                v_angle=v_angle,
                lens=lens,
            )
        else:
            return await provider.generate_angle(
                image_data=image_data,
                h_angle=h_angle,
                v_angle=v_angle,
                lens=lens,
            )

    async def generate_all_angles_stream(
        self,
        image_data: bytes,
        lens: str = "normal",
    ):
        """Generate all 9 angles, yielding results as they complete.

        Yields dicts with type='result' or type='error'.
        """
        self._init_providers()

        image_hash = compute_image_hash(image_data)
        optimized = optimize_image(image_data)

        total = len(PREDEFINED_ANGLES)
        completed = 0

        # Create tasks for all angles
        pending_tasks: dict[asyncio.Task, dict] = {}
        for angle in PREDEFINED_ANGLES:
            task = asyncio.create_task(
                self.generate_angle(
                    image_data=optimized,
                    image_hash=image_hash,
                    h_angle=float(angle["h"]),
                    v_angle=float(angle["v"]),
                    lens=lens,
                )
            )
            pending_tasks[task] = angle

        # Yield results as they complete
        while pending_tasks:
            done, _ = await asyncio.wait(
                pending_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in done:
                angle = pending_tasks.pop(task)
                completed += 1
                try:
                    img_bytes, content_type = task.result()
                    b64 = base64.b64encode(img_bytes).decode("utf-8")
                    yield {
                        "type": "result",
                        "name": angle["name"],
                        "success": True,
                        "image_data": b64,
                        "content_type": content_type,
                        "completed": completed,
                        "total": total,
                    }
                except Exception as e:
                    yield {
                        "type": "result",
                        "name": angle["name"],
                        "success": False,
                        "error": str(e),
                        "completed": completed,
                        "total": total,
                    }

    def get_active_providers(self) -> list[str]:
        """Return list of active provider names."""
        self._init_providers()
        return [p.name for p in self._providers]


# Singleton
provider_manager = ProviderManager()
