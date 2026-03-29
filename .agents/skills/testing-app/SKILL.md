# Testing AI AngleCam Nadir

## Overview
This app generates multi-angle AI images from a single input image. It uses two generation sources:
- **AngleChanger.ai** (primary) - more stable, uses proxy rewrites
- **HuggingFace/Qwen** (secondary/fallback) - can be unreliable

## Local Development Setup

### Start the dev server
```bash
cd frontend && npm install && npm run dev -- --host 0.0.0.0
```
The dev server runs on port 5173 (or next available port like 5174).

### Proxy Configuration for Local Testing
The Vercel rewrites (`/acapi` → `anglechanger.ai/api`, `/acimg` → `anglechanger.ai`) only work in production. For local testing, temporarily add proxy config to `vite.config.ts`:

```typescript
server: {
  proxy: {
    "/acapi": {
      target: "https://anglechanger.ai",
      changeOrigin: true,
      rewrite: (p) => p.replace(/^\/acapi/, "/api"),
    },
    "/acimg": {
      target: "https://anglechanger.ai",
      changeOrigin: true,
      rewrite: (p) => p.replace(/^\/acimg/, ""),
    },
  },
}
```

**Important:** Revert this change after testing - do NOT commit it.

### Vercel Preview Deployment
The Vercel preview may have deployment protection requiring login. If you can't access the preview URL, test locally with the Vite proxy config above.

## Key Testing Flows

### 1. Source Selector UI
- Three buttons: "Auto" (default), "AngleChanger", "HuggingFace"
- Each shows a description text explaining the behavior
- Auto: "Tries AngleChanger.ai first, falls back to HuggingFace"
- AngleChanger: "Uses AngleChanger.ai only (more stable)"
- HuggingFace: "Uses HuggingFace Qwen model only"

### 2. Image Generation with AngleChanger
- Upload an image → select AngleChanger source → select image count (3 is fastest for testing) → click Generate
- Progress shows "via AngleChanger.ai" indicator
- Network tab should show: `upload.php` (POST), `generate.php` (POST), `check-status.php` (GET polling), `gen_*.jpg` (result images)
- All should return 200

### 3. Auto-Fallback Testing
- In Auto mode, AngleChanger is tried first
- If AngleChanger fails, console shows `[AUTO] Falling back to HuggingFace for {angle}`
- Active source changes to "HuggingFace (fallback)"

### 4. HuggingFace-Only Regression
- Select HuggingFace source → Generate
- Network should show NO `/acapi/*` requests
- Only HuggingFace `maybe_infer` / `upload` calls
- HuggingFace may be unreliable - failures are expected and handled gracefully

## Test Image
Any PNG/JPEG/WebP image works. Can generate a simple test image with:
```python
from PIL import Image
img = Image.new('RGB', (512, 512), color=(100, 150, 200))
img.save('test-image.png')
```

## Build & Lint
```bash
cd frontend && npm run lint && npm run build
```

## Key Files
- `frontend/src/App.tsx` - Main app component with all generation logic
- `vercel.json` - Vercel rewrites for AngleChanger.ai proxy
- `frontend/vite.config.ts` - Vite build config (add proxy here for local testing)

## Devin Secrets Needed
No secrets required - both AngleChanger.ai and HuggingFace APIs work without authentication.
