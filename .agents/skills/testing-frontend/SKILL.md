# Testing AI AngleCam Nadir Frontend

## Project Structure
- Monorepo with `frontend/` (Vite + React + TypeScript) and `backend/` (FastAPI + Poetry)
- The frontend calls the HuggingFace Gradio API directly (no backend dependency for the deployed site)
- Deployed on Vercel with config in `vercel.json` at repo root
- Default branch: `devin/1774108124-multi-angle-generator`

## Local Build & Test
```bash
cd frontend
npm install
npm run build    # tsc -b && vite build
npm run lint     # eslint
npm run dev      # local dev server
```

## Vercel Configuration
- `vercel.json` at repo root configures:
  - `installCommand`: `cd frontend && npm install`
  - `buildCommand`: `cd frontend && npm run build`
  - `outputDirectory`: `frontend/dist`
  - `framework`: `vite`
  - SPA rewrites for non-asset routes
  - Cache headers for static assets

## Testing the Deployment
1. Production URL: https://aianglecam-nadir.vercel.app/
2. Preview deployments may require Vercel authentication - check the Vercel bot comment on PRs for preview URLs
3. No CI/CD pipelines configured beyond Vercel auto-deploy

## End-to-End Testing Flow
1. Navigate to the site root
2. Verify all UI sections render: header, Input Image, Lens Type, Number of Images, Generate button, Generated Angles, footer
3. Upload a test image (JPEG/PNG/WebP, max 20MB)
4. Select number of images (3-9)
5. Click "Generate N Angles" - button should be enabled only when both image AND count are selected
6. Wait for generation (~30s for 3 images) - progress indicator shows "Generating... (X/N)"
7. Verify generated images appear in the grid with success count
8. Test static asset serving: navigate to `/vite.svg` directly - should serve the SVG, not HTML
9. Test SPA routing: navigate to a non-existent path - should load the app (not 404) after SPA rewrites are configured

## Key Things to Verify
- The HuggingFace Gradio API (`linoyts-qwen-image-edit-angles.hf.space`) might be slow or down - the app has built-in retry logic (up to 4 retries per angle)
- CORS is handled by the HF Space allowing cross-origin requests
- No backend deployment needed - frontend is fully self-contained
- The app uses `tailwindcss-animate` plugin via dynamic import in `tailwind.config.js`

## Devin Secrets Needed
None - the app uses the public HuggingFace Gradio API without authentication.
