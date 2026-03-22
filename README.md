# AI AngleCam Nadir

Multi-Angle Image Generator — Upload an image and generate multiple viewing angles automatically.

**Live Site:** [https://aianglecam-nadir.vercel.app/](https://aianglecam-nadir.vercel.app/)

## Features

- Upload any image (JPEG, PNG, WebP)
- Generate up to 9 different viewing angles (Front, Front Right, Right, Back Right, Back, Back Left, Left, Front Left, Top View)
- Two generation providers:
  - **HF Space** — Free, powered by Qwen Image Edit on HuggingFace
  - **AngleChanger.ai** — Bot integration for angle-changed image generation
- Retry individual failed angles or all failed at once
- Download generated images individually or as a batch
- Responsive, modern dark UI

## Tech Stack

- **Frontend:** React 18 + TypeScript + Vite + Tailwind CSS
- **Backend:** FastAPI (Python 3.12+)
- **Deployment:** Vercel

## Getting Started

### Frontend

```bash
cd frontend
npm install
npm run dev
```

### Backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## License

MIT
