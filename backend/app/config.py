import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from backend root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://linoyts-qwen-image-edit-angles.hf.space")

# Multi-token fallback: comma-separated list of HF tokens
# Falls back to next token when current one is rate-limited or fails
HF_API_TOKENS: list[str] = [
    t.strip() for t in os.getenv("HF_API_TOKENS", HF_API_TOKEN).split(",") if t.strip()
]

# Generation defaults
DEFAULT_GUIDANCE_SCALE = 1.0
DEFAULT_INFERENCE_STEPS = 4
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0  # seconds
RETRY_MAX_DELAY = 30.0  # seconds

# Concurrency control - how many parallel generations at once
MAX_CONCURRENT_GENERATIONS = 3

# Queue settings
MAX_QUEUE_SIZE = 10  # max pending jobs in queue

# Cache settings
CACHE_MAX_SIZE = 200  # max cached results
CACHE_TTL_SECONDS = 3600  # 1 hour

# Upload settings
MAX_UPLOAD_SIZE_MB = 20
UPLOAD_DIR = "/tmp/anglecam_uploads"

# GitHub RAW image storage
GITHUB_RAW_REPO = os.getenv("GITHUB_RAW_REPO", "")  # e.g., "owner/repo"
GITHUB_RAW_TOKEN = os.getenv("GITHUB_RAW_TOKEN", "")
GITHUB_RAW_BRANCH = os.getenv("GITHUB_RAW_BRANCH", "generated-images")

# Replicate API (secondary fallback)
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN", "r8_dummy_replicate_token")
REPLICATE_MODEL = os.getenv(
    "REPLICATE_MODEL",
    "stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
)

# Stable Horde API (tertiary fallback)
STABLE_HORDE_API_KEY = os.getenv("STABLE_HORDE_API_KEY", "0000000000")
STABLE_HORDE_API_URL = "https://stablehorde.net/api/v2"
STABLE_HORDE_MODEL = os.getenv("STABLE_HORDE_MODEL", "stable_diffusion")

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
