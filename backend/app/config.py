import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from backend root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "https://linoyts-qwen-image-edit-angles.hf.space")

# Multi-token rotation: comma-separated HF tokens for massive quota boost
# Users can set HF_API_TOKENS="token1,token2,token3,..." in .env
HF_API_TOKENS_RAW = os.getenv("HF_API_TOKENS", "")
HF_API_TOKENS: list[str] = [
    t.strip() for t in HF_API_TOKENS_RAW.split(",") if t.strip()
]
# Always include the primary token if set
if HF_API_TOKEN and HF_API_TOKEN not in HF_API_TOKENS:
    HF_API_TOKENS.insert(0, HF_API_TOKEN)

# Fallback Gradio spaces for when primary space is rate-limited
# These are public HF Spaces that offer similar image editing capabilities
FALLBACK_SPACE_URLS_RAW = os.getenv("FALLBACK_SPACE_URLS", "")
FALLBACK_SPACE_URLS: list[str] = [
    u.strip() for u in FALLBACK_SPACE_URLS_RAW.split(",") if u.strip()
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

# Cache settings
CACHE_MAX_SIZE = 200  # max cached results
CACHE_TTL_SECONDS = 3600  # 1 hour

# Upload settings
MAX_UPLOAD_SIZE_MB = 20
UPLOAD_DIR = "/tmp/anglecam_uploads"

# The 9 predefined camera angles
# Each angle has distinct h (horizontal rotation), v (vertical tilt),
# and optional forward override to produce truly different images.
# h range: the Gradio API accepts full rotation values
# v range: converted to [-1, 1] by dividing by 60
PREDEFINED_ANGLES = [
    {"name": "Front", "h": 0, "v": 0, "forward": None},
    {"name": "Front Right", "h": 55, "v": 10, "forward": None},
    {"name": "Right", "h": 90, "v": 0, "forward": None},
    {"name": "Back Right", "h": 90, "v": 40, "forward": 3.5},
    {"name": "Back", "h": 0, "v": -50, "forward": 4.0},
    {"name": "Back Left", "h": -90, "v": 40, "forward": 3.5},
    {"name": "Left", "h": -90, "v": 0, "forward": None},
    {"name": "Front Left", "h": -55, "v": 10, "forward": None},
    {"name": "Top View", "h": 0, "v": 60, "forward": 1.0},
]
