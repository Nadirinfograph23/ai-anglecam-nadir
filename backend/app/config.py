import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from backend root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://linoyts-qwen-image-edit-angles.hf.space")

# Generation defaults
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_INFERENCE_STEPS = 4
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds
RETRY_MAX_DELAY = 30.0  # seconds

# Quota-specific retry configuration
QUOTA_RETRY_BASE_DELAY = 10.0  # seconds - longer wait for quota errors
QUOTA_RETRY_MAX_DELAY = 120.0  # seconds

# Rate limiting - control request frequency to HF Space
INTER_REQUEST_DELAY = 1.5  # seconds between requests
RATE_LIMIT_TOKENS = 2  # max concurrent tokens
RATE_LIMIT_REFILL_RATE = 0.5  # tokens per second

# Concurrency control - how many parallel generations at once
MAX_CONCURRENT_GENERATIONS = 2

# Cache settings
CACHE_MAX_SIZE = 200  # max cached results
CACHE_TTL_SECONDS = 3600  # 1 hour

# Upload settings
MAX_UPLOAD_SIZE_MB = 20
UPLOAD_DIR = "/tmp/anglecam_uploads"

# The 9 predefined camera angles
# NOTE: The HF Space model supports rotation from -90 to 90 degrees
# and vertical tilt from -1 to 1 (mapped from -60 to 60 degrees).
# All angles must stay within these ranges to avoid clamping/duplicates.
PREDEFINED_ANGLES = [
    {"name": "Front", "h": 0, "v": 0},
    {"name": "Front Right", "h": 45, "v": 0},
    {"name": "Right", "h": 90, "v": 0},
    {"name": "Front Left", "h": -45, "v": 0},
    {"name": "Left", "h": -90, "v": 0},
    {"name": "Top Front", "h": 0, "v": 60},
    {"name": "Top Right", "h": 45, "v": 30},
    {"name": "Top Left", "h": -45, "v": 30},
    {"name": "Low Front", "h": 0, "v": -30},
]
