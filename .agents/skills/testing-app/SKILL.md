# Testing AI AngleCam Nadir

## Architecture
- **Frontend**: React + Vite + TypeScript + Tailwind CSS (in `frontend/`)
- **Backend**: FastAPI + Python (in `backend/`), uses Poetry for dependency management
- **API**: Calls HuggingFace Gradio Space (`linoyts-qwen-image-edit-angles.hf.space`) for image generation
- **Deployment**: Frontend on Vercel, backend separately hosted

## Local Setup

### Frontend
```bash
cd frontend
npm install
VITE_API_URL="http://localhost:8000" npx vite --port 5173 --host
```

### Backend
```bash
cd backend
poetry install
# Create .env file with HF_API_TOKEN
uvicorn app.main:app --reload --port 8000
```

## Devin Secrets Needed
- `HF_API_TOKEN`: HuggingFace API token required for the backend to call the Gradio Space API for image generation. Without this, all image generation requests will fail.

## Testing Notes

### What Can Be Tested Without Backend
- Browser tab title
- Image upload and preview
- UI layout and styling
- Error handling when backend is unavailable (shows "Failed to fetch" error)

### What Requires Working Backend (HF_API_TOKEN)
- Actual image generation (9 angle views)
- Partial failure scenario (Arabic message: "تم توليد X صور من أصل Y بسبب الضغط على الخادم")
- Auto-retry logic for failed angles
- "Retry Failed" and "Retry All Failed" button functionality
- Download generated images
- Streaming progress updates

### Key Test Scenarios
1. **Full success**: Upload image → Generate → All 9 angles succeed → Download all
2. **Partial failure**: Upload image → Generate → Some angles fail → Arabic message appears → Retry Failed button works
3. **Complete failure**: No backend → "Failed to fetch" error → No broken image placeholders in grid
4. **Lens types**: Test Normal, Wide Angle, Close-Up lens options affect generation

### Important Code Paths
- `frontend/src/App.tsx`: Main component with all generation, retry, and UI logic
- `backend/app/main.py`: API endpoints (`/api/generate-stream`, `/api/retry-angle`)
- `backend/app/services/hf_client.py`: HuggingFace Space API client with retry/cache logic
- `backend/app/config.py`: Configuration (retry counts, concurrency limits, angle definitions)

### Known Issues
- The HF Space API may have rate limits causing some angle generations to fail under load
- `clamp_rotate` in `hf_client.py` clamps angles > 90° to 90° (Back/Back Right/Back Left angles produce similar results to Side views)
- Vercel deploys from the base branch, so PR changes won't be visible on the Vercel preview until merged
