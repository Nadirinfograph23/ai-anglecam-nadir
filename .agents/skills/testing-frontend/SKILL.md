# Testing AI AngleCam Nadir Frontend

## Setup

1. Install dependencies:
   ```bash
   cd frontend && npm install
   ```

2. Start the dev server:
   ```bash
   npx vite --host 0.0.0.0 --port 5173
   ```
   Note: If port 5173 is in use, Vite will auto-select the next available port (e.g., 5174). Check the terminal output for the actual URL.

3. Open the app in the browser at the URL shown in the terminal.

## Key Testing Notes

- **No backend needed**: The frontend calls the HuggingFace Gradio Space API directly (configured via `HF_SPACE_URL` constant in `App.tsx`). No backend server setup is required for testing.
- **No authentication needed**: The app is fully public with no login required.
- **Test images**: A simple solid-color PNG image (512x512) works for testing. The HF API will generate camera angle variations from it. Create one with:
  ```python
  from PIL import Image
  img = Image.new('RGB', (512, 512), color=(100, 150, 200))
  img.save('/tmp/test_image.png')
  ```
- **HF API reliability**: The HuggingFace Space API may be under load and return errors for some angles. This is expected behavior - the app handles it with retry logic and Retry buttons. Some angles (especially "Right" at 90 degrees) may fail more often under heavy load.
- **Generation time**: Each angle takes ~10-20 seconds to generate. A full 9-angle generation can take 1-3 minutes depending on API load.
- **Vercel preview**: The Vercel preview deployment might require Vercel login. Testing locally with `npm run dev` is more reliable.

## What to Test

1. **Image upload**: Upload via click or drag-and-drop
2. **Image count selector**: Verify buttons 3-9 work, warning shown when none selected, generate button disabled until count chosen
3. **Generation**: Verify correct number of grid slots appear, progress bar updates, images render without text labels
4. **Retry on failure**: Verify Retry button is centered in failed tiles, responds quickly (uses cached upload path)
5. **Download**: Verify individual image download on hover and "Download All" button

## Build & Lint

```bash
cd frontend
npm run build    # TypeScript check + Vite build
npm run lint     # ESLint
```

## Devin Secrets Needed

None required - the app uses a public HuggingFace Space API.
