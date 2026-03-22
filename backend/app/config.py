import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from backend root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

# --- Provider API Keys ---
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://linoyts-qwen-image-edit-angles.hf.space")

FAL_API_KEY = os.getenv("FAL_API_KEY", "")
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "")
STABLE_HORDE_API_KEY = os.getenv("STABLE_HORDE_API_KEY", "0000000000")  # anon key works

# Provider priority order (comma-separated). First available provider is tried first.
# Options: fal, hf, replicate, stablehorde
PROVIDER_ORDER = os.getenv("PROVIDER_ORDER", "fal,hf,replicate,stablehorde")

# Generation defaults
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_INFERENCE_STEPS = 4
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds
RETRY_MAX_DELAY = 30.0  # seconds

# Per-angle timeout (seconds) - prevents hanging indefinitely
ANGLE_TIMEOUT = 120

# Concurrency control - how many parallel generations at once
MAX_CONCURRENT_GENERATIONS = 3

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
