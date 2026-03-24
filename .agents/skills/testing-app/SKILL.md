# Testing AI AngleCam Nadir

## Overview
FastAPI backend + React/Vite frontend for multi-angle image generation using HuggingFace, Replicate, and Stable Horde APIs.

## Local Setup

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
VITE_API_URL="http://localhost:8000" npm run dev -- --host 0.0.0.0 --port 5173
```
Note: If port 5173 is in use, Vite auto-assigns a different port (e.g. 5174). Check terminal output.

## Devin Secrets Needed
- `HF_API_TOKEN` — HuggingFace API token (optional, generation works via Stable Horde fallback without it)
- `REPLICATE_API_TOKEN` — Replicate API token (optional, disabled with dummy token)
- `STABLE_HORDE_API_KEY` — Stable Horde key (optional, anonymous key `0000000000` works but may be slow/rate-limited)

## Key Endpoints
- `GET /api/status` — Shows cache stats, queue depth, token count, last API used, fallback API availability
- `POST /api/generate-stream` — SSE streaming endpoint for 9-angle generation
- `GET /healthz` — Health check

## Testing the Fallback Chain
1. Start backend WITHOUT `HF_API_TOKEN` set
2. Check `/api/status` — should show `fallback_apis.replicate: false`, `fallback_apis.stable_horde: true`
3. Upload image and click Generate
4. Monitor backend logs for:
   - `[huggingface] Attempt 1/3 failed` through `3/3 failed`
   - `[fallback] Trying Stable Horde for rotate=X.X`
   - `[stable_horde] Generation succeeded` or failure messages
5. Frontend should show per-angle progress with spinners, elapsed timer, and estimated remaining time
6. Failed angles show "Generation failed on all APIs after..." with Retry button
7. After completion, `/api/status` shows `last_api_used: stable_horde`

## Known Testing Behaviors
- Without HF tokens, HF upload fails first, then each angle goes through 3 HF API retries (returning null errors) before falling to Stable Horde
- Stable Horde with anonymous key can be rate-limited (429) when sending many concurrent requests — some angles may fail while others succeed
- The fallback chain adds significant delay (~15-30s per angle for HF retries + Horde polling)
- Replicate is skipped entirely when using the dummy token `r8_dummy_replicate_token`
- `crypto.subtle.digest` requires HTTPS (localhost is exempt) — SHA-256 hashing for localStorage cache keys

## Frontend Features to Verify
- "How it works" step-by-step guide in empty state (3 numbered steps)
- Elapsed timer and estimated remaining time during generation
- 9-angle grid with per-angle spinners
- Error banner with per-angle Retry buttons on failure
- Before/after comparison modal (eye icon overlay on generated images, requires at least 1 successful generation)
- "Download All" button appears after generation completes
- Generate button re-enables after generation finishes (success or failure)

## Lint & Build
```bash
# Frontend
cd frontend && npx eslint .
cd frontend && npx tsc --noEmit

# Backend — no formal linter configured; Python import checks via uvicorn startup
```
