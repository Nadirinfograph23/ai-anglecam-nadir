# Testing AI AngleCam Nadir Frontend

## Dev Server Setup

```bash
cd frontend
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

The frontend runs at `http://localhost:5173`. No backend is required for testing Puter.js integration or UI features - the backend is only needed for the fallback API path.

## Lint & Typecheck

```bash
cd frontend
npm run lint
npx tsc -b
```

Both must pass before creating PRs.

## Testing Puter.js Integration

### Verify SDK Loading
Open browser DevTools console and run:
```js
console.log(typeof puter, typeof puter.ai, typeof puter.ai.txt2img)
// Expected: "object" "object" "function"
```

### Known Issue: Puter.js Auth Popup
Puter.js opens a Cloudflare verification popup on first use. In automated testing environments, this verification may fail. When it fails, the app should automatically fall back to the Backend API after the first 2 angle timeouts (each angle has a 60-second timeout).

### Testing the Generation Flow
1. Upload any image (JPEG, PNG, WebP, max 20MB)
2. Click "Generate All 9 Angles"
3. Observe "Generating via: Puter.js AI" indicator
4. If Puter.js auth fails, wait ~2 minutes for fallback to Backend API
5. Without a backend running, the fallback will show "Failed to fetch"

## Testing Daily Usage Limits

Usage is tracked in localStorage with keys like `anglecam_usage_YYYY-MM-DD`.

### Test Full Quota (5/5 remaining)
```js
localStorage.removeItem('anglecam_usage_' + new Date().toISOString().split('T')[0])
// Reload page - should show "5/5 remaining" with cyan progress bar
```

### Test Low Quota - Yellow Warning (1-2 remaining)
```js
localStorage.setItem('anglecam_usage_' + new Date().toISOString().split('T')[0], '3')
// Reload page - should show "2/5 remaining" with yellow progress bar
```

### Test Limit Reached (0 remaining)
```js
localStorage.setItem('anglecam_usage_' + new Date().toISOString().split('T')[0], '5')
// Reload page - should show:
// - "0/5 remaining" in red
// - "Daily Limit Reached" on button (disabled)
// - Red warning text below progress bar
```

Note: The usage indicator auto-refreshes every 30 seconds via a React interval, but reloading the page is faster for testing.

## Test Image

A simple test image can be generated with Python:
```python
from PIL import Image, ImageDraw
img = Image.new('RGB', (400, 400), color='white')
draw = ImageDraw.Draw(img)
draw.polygon([(150,200),(200,150),(250,200),(200,250)], fill='red', outline='black')
img.save('/tmp/test-product.png')
```

## Key UI Elements to Verify

- Header: "Powered by Puter.js AI + Qwen Image Edit"
- Upload area with drag-and-drop support
- Lens type selector (Normal, Wide Angle, Close-Up)
- Generate button with state changes (enabled/generating/limit reached)
- Daily Usage indicator with color-coded progress bar
- API source indicator during generation
- 9-angle grid with loading spinners, results, or error+retry buttons
- Download individual/all buttons on successful results

## Architecture Notes

- Frontend: React 18 + TypeScript + Vite + Tailwind CSS
- Puter.js SDK loaded via script tag in index.html
- Generation flow: Puter.js first -> Backend API fallback
- Fallback triggers if first 2 Puter.js angles fail
- Backend API uses SSE (Server-Sent Events) for streaming progress
- Usage limits are client-side only (localStorage) - can be bypassed
