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
MAX_RETRIES = 4
RETRY_BASE_DELAY = 3.0  # seconds
RETRY_MAX_DELAY = 60.0  # seconds

# Quota-specific retry settings
QUOTA_RETRY_BASE_DELAY = 10.0  # longer initial delay for quota errors
QUOTA_RETRY_MAX_DELAY = 120.0  # longer max delay for quota errors

# Concurrency control - how many parallel generations at once
MAX_CONCURRENT_GENERATIONS = 2

# Delay between sequential API calls to avoid bursting
INTER_REQUEST_DELAY = 1.5  # seconds between API calls

# Global rate limiting (requests per minute)
GLOBAL_RATE_LIMIT_RPM = 20

# Cache settings
CACHE_MAX_SIZE = 200  # max cached results
CACHE_TTL_SECONDS = 3600  # 1 hour

# Upload settings
MAX_UPLOAD_SIZE_MB = 20
UPLOAD_DIR = "/tmp/anglecam_uploads"

# The 9 predefined camera angles
PREDEFINED_ANGLES = [
    {"name": "Front", "h": 0, "v": 0},
    {"name": "Front Right", "h": 45, "v": 0},
    {"name": "Right", "h": 90, "v": 0},
    {"name": "Back Right", "h": 135, "v": 0},
    {"name": "Back", "h": 180, "v": 0},
    {"name": "Back Left", "h": -135, "v": 0},
    {"name": "Left", "h": -90, "v": 0},
    {"name": "Front Left", "h": -45, "v": 0},
    {"name": "Top View", "h": 0, "v": 60},
]
