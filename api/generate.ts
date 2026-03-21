import type { VercelRequest, VercelResponse } from "@vercel/node";

/**
 * Vercel Serverless API for generating camera angles via HuggingFace Space.
 *
 * Features:
 * - Server-side multi-token rotation for unlimited quota
 * - Automatic retry with exponential backoff
 * - Rate-limit detection and token cooldown
 * - Fallback across multiple HF Space instances
 * - Request queuing via Vercel's auto-scaling
 */

const HF_SPACE_URL =
  process.env.HF_SPACE_URL ||
  "https://linoyts-qwen-image-edit-angles.hf.space";

// Multi-token support: comma-separated HF tokens
const HF_API_TOKENS: string[] = (() => {
  const raw = process.env.HF_API_TOKENS || process.env.HF_API_TOKEN || "";
  return raw
    .split(",")
    .map((t) => t.trim())
    .filter(Boolean);
})();

// Fallback spaces for when primary is overloaded
const FALLBACK_SPACES: string[] = (() => {
  const raw = process.env.FALLBACK_SPACE_URLS || "";
  return raw
    .split(",")
    .map((u) => u.trim())
    .filter(Boolean);
})();

const ALL_SPACES = [HF_SPACE_URL, ...FALLBACK_SPACES];

// Generation defaults
const DEFAULT_GUIDANCE_SCALE = 1.0;
const DEFAULT_INFERENCE_STEPS = 4;
const DEFAULT_WIDTH = 1024;
const DEFAULT_HEIGHT = 1024;
const MAX_RETRIES = 4;
const RETRY_BASE_DELAY = 2000; // ms

// Token rotation state (persists across warm invocations)
let tokenIndex = 0;
const tokenCooldowns: Map<string, number> = new Map();
const COOLDOWN_MS = 60_000;

function getNextToken(): string {
  if (HF_API_TOKENS.length === 0) return "";
  const now = Date.now();

  // Try to find a non-cooled-down token
  for (let i = 0; i < HF_API_TOKENS.length; i++) {
    const idx = (tokenIndex + i) % HF_API_TOKENS.length;
    const token = HF_API_TOKENS[idx];
    const cooldownUntil = tokenCooldowns.get(token) || 0;
    if (now >= cooldownUntil) {
      tokenIndex = (idx + 1) % HF_API_TOKENS.length;
      return token;
    }
  }

  // All tokens cooled down, use the one with earliest cooldown
  let earliest = HF_API_TOKENS[0];
  let earliestTime = tokenCooldowns.get(earliest) || 0;
  for (const t of HF_API_TOKENS) {
    const cd = tokenCooldowns.get(t) || 0;
    if (cd < earliestTime) {
      earliest = t;
      earliestTime = cd;
    }
  }
  return earliest;
}

function markTokenRateLimited(token: string): void {
  tokenCooldowns.set(token, Date.now() + COOLDOWN_MS);
}

function clearTokenCooldown(token: string): void {
  tokenCooldowns.delete(token);
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

interface GenerateParams {
  imageBase64: string;
  rotateDeg: number;
  moveForward: number;
  verticalTilt: number;
  wideangle: boolean;
  seed?: number;
  randomizeSeed?: boolean;
}

async function uploadImage(
  imageBuffer: Buffer,
  spaceUrl: string,
  token: string
): Promise<string> {
  const boundary = "----FormBoundary" + Math.random().toString(36).slice(2);
  const filename = "input.png";

  const header = `--${boundary}\r\nContent-Disposition: form-data; name="files"; filename="${filename}"\r\nContent-Type: image/png\r\n\r\n`;
  const footer = `\r\n--${boundary}--\r\n`;

  const headerBuf = Buffer.from(header);
  const footerBuf = Buffer.from(footer);
  const body = Buffer.concat([headerBuf, imageBuffer, footerBuf]);

  const headers: Record<string, string> = {
    "Content-Type": `multipart/form-data; boundary=${boundary}`,
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${spaceUrl}/gradio_api/upload`, {
    method: "POST",
    headers,
    body,
  });

  if (response.status === 429) {
    throw new Error("429_RATE_LIMITED");
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(
      `Upload failed: ${response.status} - ${text.slice(0, 200)}`
    );
  }

  const result = await response.json();
  if (Array.isArray(result) && result.length > 0) {
    return result[0];
  }
  throw new Error("Unexpected upload response: " + JSON.stringify(result));
}

async function submitJob(
  spaceUrl: string,
  uploadedPath: string,
  params: GenerateParams,
  token: string
): Promise<string> {
  const payload = {
    data: [
      false, // is_reset_val
      { path: uploadedPath, meta: { _type: "gradio.FileData" } },
      params.rotateDeg,
      params.moveForward,
      params.verticalTilt,
      params.wideangle,
      params.seed ?? 0,
      params.randomizeSeed ?? true,
      DEFAULT_GUIDANCE_SCALE,
      DEFAULT_INFERENCE_STEPS,
      DEFAULT_WIDTH,
      DEFAULT_HEIGHT,
      null, // prev_output
    ],
  };

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(
    `${spaceUrl}/gradio_api/call/maybe_infer`,
    {
      method: "POST",
      headers,
      body: JSON.stringify(payload),
    }
  );

  if (response.status === 429) {
    throw new Error("429_RATE_LIMITED");
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(
      `Submit failed: ${response.status} - ${text.slice(0, 300)}`
    );
  }

  const data = await response.json();
  const eventId = data.event_id;
  if (!eventId) {
    throw new Error("No event_id in submit response");
  }
  return eventId;
}

async function pollResult(
  spaceUrl: string,
  eventId: string,
  token: string
): Promise<string> {
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(
    `${spaceUrl}/gradio_api/call/maybe_infer/${eventId}`,
    { headers }
  );

  if (response.status === 429) {
    throw new Error("429_RATE_LIMITED");
  }

  if (!response.ok) {
    throw new Error(`Result fetch failed: ${response.status}`);
  }

  const text = await response.text();
  return parseSSEResponse(text);
}

function parseSSEResponse(text: string): string {
  const lines = text.split("\n");
  let errorMsg = "";

  for (const line of lines) {
    if (line.startsWith("event: error")) {
      errorMsg = "API returned an error";
    }
    if (line.startsWith("data: ")) {
      const dataStr = line.slice(6).trim();
      if (dataStr === "null") continue;
      try {
        const data = JSON.parse(dataStr);
        if (typeof data === "object" && data !== null && !Array.isArray(data) && "error" in data) {
          errorMsg = String(data.error);
          continue;
        }
        if (Array.isArray(data) && data.length > 0) {
          const first = data[0];
          if (first && typeof first === "object" && "url" in first) {
            return first.url;
          }
        }
      } catch {
        continue;
      }
    }
  }

  throw new Error(errorMsg || "No result image found in SSE response");
}

async function downloadImage(
  imageUrl: string,
  token: string
): Promise<{ data: Buffer; contentType: string }> {
  const headers: Record<string, string> = {};
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(imageUrl, { headers });

  if (!response.ok) {
    throw new Error(`Image download failed: ${response.status}`);
  }

  const contentType =
    response.headers.get("content-type") || "image/webp";
  const arrayBuffer = await response.arrayBuffer();
  return { data: Buffer.from(arrayBuffer), contentType };
}

async function generateAngle(
  params: GenerateParams
): Promise<{ imageBase64: string; contentType: string }> {
  const imageBuffer = Buffer.from(params.imageBase64, "base64");

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const token = getNextToken();
    const spaceUrl = ALL_SPACES[attempt % ALL_SPACES.length];

    try {
      // Step 1: Upload
      const uploadedPath = await uploadImage(imageBuffer, spaceUrl, token);

      // Step 2: Submit job
      const eventId = await submitJob(spaceUrl, uploadedPath, params, token);

      // Step 3: Poll for result
      const imageUrl = await pollResult(spaceUrl, eventId, token);

      // Step 4: Download generated image
      const result = await downloadImage(imageUrl, token);

      // Success - clear cooldown
      clearTokenCooldown(token);

      return {
        imageBase64: result.data.toString("base64"),
        contentType: result.contentType,
      };
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : String(e);

      if (errMsg.includes("429") || errMsg.includes("rate") || errMsg.includes("quota")) {
        markTokenRateLimited(token);
      }

      console.warn(
        `[generate] Attempt ${attempt + 1}/${MAX_RETRIES} failed (space=${spaceUrl}): ${errMsg}`
      );

      if (attempt < MAX_RETRIES - 1) {
        const delay =
          RETRY_BASE_DELAY * Math.pow(1.5, attempt) +
          Math.random() * 2000;
        await sleep(delay);
      }
    }
  }

  throw new Error("Generation failed after all retries");
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  // Handle CORS preflight
  if (req.method === "OPTIONS") {
    res.status(200).end();
    return;
  }

  if (req.method !== "POST") {
    res.status(405).json({ error: "Method not allowed" });
    return;
  }

  try {
    const {
      image_base64,
      rotate_deg = 0,
      move_forward = 2.0,
      vertical_tilt = 0,
      wideangle = false,
      seed = 0,
      randomize_seed = true,
      angle_name = "",
    } = req.body || {};

    if (!image_base64) {
      res.status(400).json({ error: "image_base64 is required" });
      return;
    }

    const result = await generateAngle({
      imageBase64: image_base64,
      rotateDeg: rotate_deg,
      moveForward: move_forward,
      verticalTilt: vertical_tilt,
      wideangle: wideangle,
      seed: seed,
      randomizeSeed: randomize_seed,
    });

    res.status(200).json({
      name: angle_name,
      success: true,
      image_data: result.imageBase64,
      content_type: result.contentType,
    });
  } catch (e) {
    const errMsg = e instanceof Error ? e.message : String(e);
    console.error("[generate] Error:", errMsg);
    res.status(500).json({
      name: req.body?.angle_name || "",
      success: false,
      error: errMsg,
    });
  }
}
