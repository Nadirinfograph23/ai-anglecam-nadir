# Testing AI AngleCam Nadir

## Overview
This app has a FastAPI backend and React/Vite frontend. The backend calls a HuggingFace Gradio Space API to generate multi-angle views of uploaded images.

## Local Setup

### Frontend
```bash
cd frontend
npm install
VITE_API_URL="http://localhost:8000" npx vite --host 0.0.0.0 --port 5173
```
Note: Vite may auto-increment the port if 5173 is in use (e.g., 5174).

### Backend
```bash
cd backend
pip install fastapi uvicorn python-multipart httpx python-dotenv
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Verify Services
- Frontend: `curl http://localhost:5173` (or whichever port Vite chose)
- Backend health: `curl http://localhost:8000/healthz` (should return `{"status":"ok"}`)
- Backend status: `curl http://localhost:8000/api/status` (returns cache/queue/token info)

## Devin Secrets Needed
- `HF_API_TOKEN` - HuggingFace API token for actual image generation. Without this, generation will fail but error handling can still be tested.
- `HF_API_TOKENS` - (Optional) Comma-separated list of multiple HF tokens for multi-token rotation testing.
- `GITHUB_RAW_REPO` - (Optional) GitHub repo in `owner/repo` format for image caching.
- `GITHUB_RAW_TOKEN` - (Optional) GitHub token with repo write access for image caching.

## Testing Without HF Tokens
Without HF API tokens, you can still test:
- Frontend UX features (step-by-step guide, compression indicator, loading states)
- Error handling (graceful failure with error banner)
- Backend status endpoint
- localStorage cache behavior (verify failed results are NOT cached)
- Image upload and compression flow
- Before/after comparison modal (requires at least one successful generation)

The generation will fail with "Failed to upload image after all retries" — this is expected.

## Testing With HF Tokens
With valid HF tokens, you can additionally test:
- Full 9-angle generation flow
- SSE streaming progress updates
- Elapsed timer incrementing during generation
- localStorage caching of successful results
- Before/after comparison modal with generated images
- Download individual and Download All functionality
- Retry button on failed angles
- Multi-token rotation (set multiple tokens in HF_API_TOKENS)

## Key Test Paths

### Frontend (App.tsx)
1. **Empty state**: "HOW IT WORKS" guide with 3 numbered steps visible below "No images generated yet"
2. **Image upload**: Click "Upload Image" → file chooser → image preview appears with X button
3. **Compression**: Brief "Compressing image for faster upload..." indicator (may flash quickly for small files)
4. **Generation**: Click "Generate All 9 Angles" → 9 grid slots with spinners, progress bar, elapsed timer, estimated remaining time
5. **Error state**: Red error banner with specific message, Generate button re-enables
6. **Before/After**: Hover over generated image → Eye icon → Click to open side-by-side comparison modal

### Backend Endpoints
- `GET /healthz` - Health check
- `GET /api/angles` - List of 9 predefined angles
- `GET /api/status` - Cache stats, queue pending count, token count
- `POST /api/generate-stream` - SSE streaming generation (needs FormData with image + lens)
- `POST /api/generate-single` - Single angle generation
- `POST /api/retry-angle` - Retry a failed angle

## Known Issues
- `crypto.subtle.digest` requires HTTPS (secure context) for the image hash function. Localhost is treated as secure context by browsers, so this works locally. Production deployments on HTTPS will also work. However, plain HTTP deployments (non-localhost) might fail silently.
- The `compressImage` function creates `URL.createObjectURL` that is never revoked — minor memory leak on repeated file selections.
- The `ERR_INVALID_URL` console error for `data:image/png;base64...` may appear if base64 data is truncated. This is a cosmetic console error and doesn't affect functionality.

## Test Image Creation
If you need a test image:
```python
from PIL import Image, ImageDraw
img = Image.new('RGB', (1200, 900), (30, 60, 120))
d = ImageDraw.Draw(img)
d.rectangle([200, 150, 600, 500], fill=(200, 80, 50))
d.ellipse([500, 300, 900, 700], fill=(50, 180, 100))
img.save('/tmp/test_image.png')
```

For testing compression, use a larger image (4000x3000) so the compression indicator is visible longer.
