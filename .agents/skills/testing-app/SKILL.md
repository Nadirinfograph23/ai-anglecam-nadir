# Testing AI AngleCam Nadir

## Overview
This app generates multi-angle views of images using a HuggingFace Gradio Space API (Qwen Image Edit). It has a FastAPI backend and React/Vite frontend.

## Local Setup

### Backend
```bash
cd backend
pip install "fastapi[standard]" python-dotenv httpx pillow python-multipart aiofiles
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Note: `uvicorn` is included in `fastapi[standard]` but may not be on PATH directly. Use `python -m uvicorn` instead.

### Frontend
```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```
The frontend connects to `http://localhost:8000` by default (set via `VITE_API_URL` env var).

## Key API Endpoints
- `GET /healthz` - Health check
- `GET /api/angles` - Returns the 9 predefined camera angles with their h/v values
- `POST /api/generate-stream` - SSE streaming endpoint for generating all 9 angles
- `POST /api/retry-angle` - Retry a specific failed angle

## Known Issues

### HF Space GPU Availability
The HF Space at `linoyts-qwen-image-edit-angles.hf.space` may return `event: error` with `data: null` when the GPU backend is sleeping or unavailable. This is a Space-side issue, not a code bug. When this happens:
- All generation requests fail with "API returned an error"
- The retry logic correctly retries 3 times with exponential backoff (2s, 4s)
- The UI shows error states with Retry buttons for each angle

To verify if the Space is available, check:
```bash
curl -s "https://linoyts-qwen-image-edit-angles.hf.space/gradio_api/info" | python3 -m json.tool
```
If this returns the API info, the Space is running but the GPU may still be unavailable for inference.

### What You CAN Test Without GPU
- UI layout and angle names in the 3x3 grid
- Backend `/api/angles` returns correct angle definitions
- Retry logic and exponential backoff (visible in backend logs)
- Error handling UI (error icons, Retry buttons, success/fail counts)
- Cooldown UI behavior (only triggers for quota-specific errors, not generic API errors)
- Image upload and optimization pipeline

### What Requires Working GPU
- End-to-end image generation producing actual angle images
- Visual verification that generated images are distinct perspectives
- Quota error cooldown UI (requires actually hitting quota limits)

## Angle Definitions
The HF Space model supports:
- Rotation: -90 to 90 degrees (horizontal)
- Vertical tilt: -1 to 1 (mapped from -60 to 60 degrees)

Angles outside these ranges get clamped, producing duplicate images. All predefined angles must stay within these constraints.

## Devin Secrets Needed
- `HF_API_TOKEN` (optional) - HuggingFace API token. The app works without it (auth headers are conditional), but having one may help with rate limits on the HF Space.
