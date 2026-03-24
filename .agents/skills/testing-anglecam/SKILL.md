# Testing AI AngleCam Nadir

## Project Structure
- `frontend/` - React + Vite + Tailwind CSS app (deployed on Vercel)
- `backend/` - FastAPI Python backend using HuggingFace Gradio Space API

## Frontend Testing (No Backend Needed)

### Setup
```bash
cd frontend
npm install
npm run dev  # Starts on localhost:5173 (or next available port)
```

### Build Verification
```bash
npm run build  # Runs tsc -b && vite build
```
- Check `frontend/dist/assets/` for code-split chunks (vendor, icons, index)
- Check `frontend/dist/index.html` for meta tags, preconnect hints, critical CSS

### Lint & Type Check
```bash
npm run lint   # ESLint
npx tsc -b     # TypeScript type checking
```

### UI Testing Without Backend
- Upload an image and verify preview renders with X button to remove
- Toggle lens type buttons (Normal, Wide Angle, Close-Up) - selected has white bg
- Click "Generate All 9 Angles" without backend running to test error handling
- Verify error message appears in red box below generate button
- Verify page remains interactive after error (not frozen/crashed)

### Error Boundary Testing
- The app wraps in `ErrorBoundary` component (see `App.tsx`)
- Network failures during generation show inline error, not white screen
- Runtime React errors should show "Something went wrong" fallback UI

## Backend Testing

### Devin Secrets Needed
- `HF_API_TOKEN` - HuggingFace API token for accessing the Qwen Image Edit Space

### Setup
```bash
cd backend
poetry install
HF_API_TOKEN=<token> uvicorn app.main:app --reload
```

### Key Endpoints
- `GET /healthz` - Health check
- `GET /api/angles` - List predefined angles
- `POST /api/generate-stream` - SSE streaming generation (main endpoint)
- `POST /api/retry-angle` - Retry a single failed angle

### Important Notes
- The `clamp_rotate()` function normalizes angles to [-180, 180] range. If the HF model doesn't support angles beyond +-90, back angles (135, 180, -135) may fail. Check model compatibility.
- GZip middleware might buffer SSE responses. Test that streaming works incrementally.
- Backend uses connection pooling via httpx with 300s timeout for HF Space calls.

## Deployment
- Frontend deploys to Vercel via `vercel.json` config
- `vercel.json` sets cache headers for static assets and build configuration
- Check Vercel preview deployments on PRs for deployment verification
