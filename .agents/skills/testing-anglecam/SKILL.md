# Testing AI AngleCam Nadir

## Architecture
- **Frontend**: React + Vite + TypeScript + Tailwind CSS (runs on port 5173)
- **Backend**: FastAPI + Python (runs on port 8000)
- **External API**: HuggingFace Gradio Space at `https://linoyts-qwen-image-edit-angles.hf.space`
- **Deployment**: Frontend on Vercel, backend needs separate hosting or local run

## Local Setup

### Backend
```bash
cd backend
pip install "fastapi[standard]"
poetry install --no-root
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend defaults to `http://localhost:8000` for API URL (configured in `App.tsx` line ~16 as `VITE_API_URL`).

## Known Blockers

### HF Space GPU Quota
The HF Space uses free-tier GPU which has a daily quota (~300s). When exceeded, ALL API calls return `event: error, data: null` with no useful error message in the SSE response. The HF Space UI shows: "You have exceeded your GPU quota (Xs requested vs Ys left). Try again in HH:MM:SS".

**Workaround**: Wait for quota reset (~24h) or use a HuggingFace Pro account with higher quota.

**To check quota status**: Visit `https://linoyts-qwen-image-edit-angles.hf.space`, upload an image, move the camera slider, and click Generate. If quota is exceeded, you'll see an error banner.

### Auth Header
When `HF_API_TOKEN` is not set (empty string), the backend must NOT send `Authorization: Bearer ` header — this causes `Illegal header value` errors. The `_auth_headers()` method handles this by returning an empty dict when no token is configured.

The HF Space is public and works without authentication, so `HF_API_TOKEN` is optional.

## Testing Checklist

### What can be tested without GPU quota
- Cancel button UI (appears during generation, stops cleanly)
- Error handling (retry buttons, error messages, status counts)
- UI recovery after generation completes/fails
- Re-generation without page reload
- `clamp_rotate()` unit test via Python
- Backend log verification (correct rotation values sent)
- Auth header fix (no `Illegal header value` errors)

### What requires GPU quota
- End-to-end image generation
- Back angle image quality (rotation values >90°)
- SSE streaming success path
- Timeout/stall detection (needs long-running generation)
- Cancel during successful generation

## Key Files
- `backend/app/services/hf_client.py` - HF Space client with upload, generation, SSE streaming
- `backend/app/config.py` - Configuration including predefined angles
- `frontend/src/App.tsx` - Main React component with generation logic and cancel button

## Devin Secrets Needed
- `VERCEL_TOKEN` - For deploying frontend to Vercel (optional, for preview deployments)
- `HF_API_TOKEN` - HuggingFace API token (optional, Space is public but token may help with quota)
