# Testing AI AngleCam Nadir - Multi-Angle Image Generator

## Overview
This app generates 9 different viewing angles of an uploaded image using the HuggingFace Gradio Space API (Qwen Image Edit). The frontend calls the HF Space directly from the browser with token rotation.

## Architecture
- **Frontend-only deployment**: Static site built with Vite + React + TypeScript
- **No backend required**: Frontend calls `https://linoyts-qwen-image-edit-angles.hf.space/gradio_api/*` directly
- **Token rotation**: Multiple HF API tokens are embedded at build time via `VITE_HF_TOKEN_1` through `VITE_HF_TOKEN_9` env vars
- **Deduplication**: The `clampRotate()` function caps rotation to ±90°, so multiple angles map to the same API parameters. The app detects this and makes ~6 unique API calls instead of 9.

## Devin Secrets Needed
- `VITE_HF_TOKEN_1` through `VITE_HF_TOKEN_9`: HuggingFace API tokens for quota distribution. These are set in `frontend/.env` at build time.

## How to Build & Deploy
1. Create `frontend/.env` with all `VITE_HF_TOKEN_*` variables
2. Run `npm run build` in the `frontend/` directory
3. Deploy the `frontend/dist/` folder as a static site

## How to Test

### Setup
1. Deploy the frontend (or run `npm run dev` locally in `frontend/`)
2. Prepare a test image (any PNG/JPEG, doesn't need to be realistic)

### End-to-End Test Flow
1. Open the deployed URL in Chrome with DevTools Console open
2. Upload a test image via the "Upload Image" button
3. Keep lens on "Normal" (default)
4. Click "Generate All 9 Angles"
5. **Verify deduplication**: Console should show `"Deduplication: 9 angles -> 6 unique API calls"` for Normal lens
6. **Verify progress**: Progress bar should increment from 0/9 to 9/9
7. **Verify grid**: All 9 angle cells should populate with images (or show error+retry for failed ones)
8. **Verify retry**: If any angle fails, click "Retry" button - it should attempt generation with a different token
9. **Verify download**: Hover over a generated image to see download overlay; click to download individual image. Click "Download All" to batch download.

### Expected Behaviors
- Generation takes 30-120 seconds total (sequential API calls)
- Some angles may fail due to GPU quota limits - this is expected. The retry button uses a different HF token.
- The `clampRotate` function means Back Right, Back, and Back Left will produce similar images to Right/Left (they map to the same rotation params)
- Top View uses a vertical tilt parameter and produces a distinct aerial perspective
- Images are generated at 768x768 resolution by default

### Known Issues
- **CORS**: HuggingFace Gradio Spaces generally allow CORS, but if the Space is restarted or updated, CORS might temporarily fail
- **Quota errors**: GPU quota on HF Spaces is limited. Token rotation helps but doesn't eliminate the issue. Errors like "API returned an error" after 4 retries indicate quota exhaustion on all tokens.
- **Token security**: HF tokens are embedded in the built JS bundle. Anyone inspecting the deployed site's JS can extract them. This is a known tradeoff for the frontend-only architecture.
- **Image preview error**: The browser console may show `GET data:image/png;base64,... net::ERR_INVALID_URL` - this is a harmless preview rendering issue, not related to generation.

### Console Logs to Watch
- `"Deduplication: X angles -> Y unique API calls"` - confirms dedup is working
- `"Generation attempt N/4 failed: ..."` - shows retry logic with quota-aware backoff
- `"Generating angle: {name} (rotate=X, forward=Y, tilt=Z, wide=W)"` - shows each API call
