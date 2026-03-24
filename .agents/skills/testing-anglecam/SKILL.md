# Testing AI AngleCam Nadir

## Local Setup

### Backend
```bash
cd backend
poetry install --no-root
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

The frontend connects to `http://localhost:8000` by default (configurable via `VITE_API_URL` env var).

## Provider Configuration

The app uses a multi-provider fallback system. Providers are tried in order defined by `PROVIDER_ORDER` env var (default: `fal,hf,replicate,stablehorde`).

| Provider | Env Var | Notes |
|----------|---------|-------|
| fal.ai (primary) | `FAL_API_KEY` | Fastest, supports full 0-360 rotation. Get key at https://fal.ai/dashboard/keys |
| HuggingFace Space | `HF_API_TOKEN` | Uses public Gradio space. May fail with upload errors without token. |
| Replicate | `REPLICATE_API_TOKEN` | Uses SDXL img2img. Get token at https://replicate.com/account/api-tokens |
| Stable Horde | `STABLE_HORDE_API_KEY` | Free anonymous key works (`0000000000`) but queue is very slow (may timeout). |

## Devin Secrets Needed

- `FAL_API_KEY` - Required for the primary (fastest) provider
- `HF_API_TOKEN` - Optional, improves HF Space reliability
- `REPLICATE_API_TOKEN` - Optional fallback provider
- `STABLE_HORDE_API_KEY` - Optional, anonymous key works but is slow

At minimum, `FAL_API_KEY` should be set for meaningful end-to-end testing.

## Key API Endpoints

- `GET /healthz` - Returns status and active provider list
- `GET /api/providers` - Returns list of active provider names
- `GET /api/angles` - Returns the 9 predefined camera angles
- `POST /api/generate-single` - Generate one angle (params: image, h_angle, v_angle, lens)
- `POST /api/generate-stream` - SSE stream all 9 angles (params: image, lens)
- `POST /api/retry-angle` - Retry a failed angle (params: image, angle_name, lens)

## Testing Checklist

1. Verify `/healthz` returns correct provider list
2. Verify `/api/providers` matches expected providers based on env vars
3. Upload image in frontend - verify preview and Generate button
4. Click Generate All 9 Angles - verify SSE streaming progress (0/9 → 9/9)
5. Check backend logs for provider fallback messages
6. Verify failed angles show error message + Retry button
7. Verify successful angles show generated images in 3x3 grid

## Known Issues

- **HF Space uploads may fail**: The public Gradio space at `linoyts-qwen-image-edit-angles.hf.space` may reject uploads without an HF token. Fallback to next provider handles this gracefully.
- **Stable Horde anonymous queue is very slow**: With the default anonymous key, jobs may take >120s and timeout. This is normal for free-tier usage.
- **Rotation clamping on HF path**: The HF client still clamps rotation to +/-90 degrees. Full 0-360 rotation only works with fal.ai provider.
- **Rate limiting on Stable Horde**: Anonymous users get rate-limited to 2 requests/second. The retry logic handles this but may cause delays.

## Lint Commands

- Frontend: `cd frontend && npm run lint`
- Backend: `cd backend && ruff check .` (if ruff is installed) or `python3 -m py_compile <file>` for syntax checks
