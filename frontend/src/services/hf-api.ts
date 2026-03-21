/**
 * Direct HuggingFace Gradio Space API client with token rotation.
 * Calls the HF Space directly from the browser, no backend needed.
 */

const HF_SPACE_URL = "https://linoyts-qwen-image-edit-angles.hf.space";

// Tokens loaded from env vars at build time
const HF_TOKENS: string[] = [
  import.meta.env.VITE_HF_TOKEN_1 || "",
  import.meta.env.VITE_HF_TOKEN_2 || "",
  import.meta.env.VITE_HF_TOKEN_3 || "",
  import.meta.env.VITE_HF_TOKEN_4 || "",
  import.meta.env.VITE_HF_TOKEN_5 || "",
  import.meta.env.VITE_HF_TOKEN_6 || "",
  import.meta.env.VITE_HF_TOKEN_7 || "",
  import.meta.env.VITE_HF_TOKEN_8 || "",
  import.meta.env.VITE_HF_TOKEN_9 || "",
].filter((t) => t.length > 0);

let tokenIndex = 0;

function getNextToken(): string {
  if (HF_TOKENS.length === 0) return "";
  const token = HF_TOKENS[tokenIndex % HF_TOKENS.length];
  tokenIndex++;
  return token;
}

function authHeaders(token: string): Record<string, string> {
  if (!token) return {};
  return { Authorization: `Bearer ${token}` };
}

// Generation defaults
const DEFAULT_GUIDANCE_SCALE = 1.0;
const DEFAULT_INFERENCE_STEPS = 4;
const DEFAULT_WIDTH = 768;
const DEFAULT_HEIGHT = 768;

// Retry config
const MAX_RETRIES = 4;
const RETRY_BASE_DELAY = 3000; // ms
const QUOTA_RETRY_DELAY = 30000; // ms

export interface AngleConfig {
  name: string;
  h: number;
  v: number;
}

export const PREDEFINED_ANGLES: AngleConfig[] = [
  { name: "Front", h: 0, v: 0 },
  { name: "Front Right", h: 45, v: 0 },
  { name: "Right", h: 90, v: 0 },
  { name: "Back Right", h: 135, v: 0 },
  { name: "Back", h: 180, v: 0 },
  { name: "Back Left", h: -135, v: 0 },
  { name: "Left", h: -90, v: 0 },
  { name: "Front Left", h: -45, v: 0 },
  { name: "Top View", h: 0, v: 60 },
];

function clampRotate(deg: number): number {
  if (-90 <= deg && deg <= 90) return deg;
  if (90 < deg && deg <= 180) return 90;
  if (-180 <= deg && deg < -90) return -90;
  return 0;
}

function convertVertical(v: number): number {
  return Math.max(-1.0, Math.min(1.0, v / 60.0));
}

function convertForward(lens: string): number {
  const mapping: Record<string, number> = { closeup: 5.0, wide: 0.0, normal: 2.0 };
  return mapping[lens] ?? 2.0;
}

function isQuotaError(error: string): boolean {
  const lower = error.toLowerCase();
  const keywords = [
    "quota", "rate limit", "429", "too many requests",
    "queue is full", "exceeded", "gpu quota", "limit reached",
    "no gpu", "queue_full", "busy",
  ];
  return keywords.some((kw) => lower.includes(kw));
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/** Optimize image: resize to max 2048px and convert to PNG blob */
async function optimizeImage(file: File, maxSize = 2048): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      let w = img.width;
      let h = img.height;
      if (Math.max(w, h) > maxSize) {
        const ratio = maxSize / Math.max(w, h);
        w = Math.round(w * ratio);
        h = Math.round(h * ratio);
      }
      const canvas = document.createElement("canvas");
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) { reject(new Error("Canvas not supported")); return; }
      ctx.drawImage(img, 0, 0, w, h);
      canvas.toBlob(
        (blob) => {
          if (blob) resolve(blob);
          else reject(new Error("Failed to convert image"));
        },
        "image/png",
      );
    };
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = URL.createObjectURL(file);
  });
}

/** Upload image to HF Space, returns the file path */
async function uploadImage(imageBlob: Blob, token: string): Promise<string> {
  const formData = new FormData();
  formData.append("files", imageBlob, "input.png");

  const response = await fetch(`${HF_SPACE_URL}/gradio_api/upload`, {
    method: "POST",
    headers: authHeaders(token),
    body: formData,
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Upload failed (${response.status}): ${text.slice(0, 200)}`);
  }

  const result = await response.json();
  if (Array.isArray(result) && result.length > 0) {
    return result[0];
  }
  throw new Error(`Unexpected upload response: ${JSON.stringify(result)}`);
}

/** Submit a generation job and return the event_id */
async function submitJob(
  uploadedPath: string,
  rotateDeg: number,
  moveForward: number,
  verticalTilt: number,
  wideangle: boolean,
  token: string,
): Promise<string> {
  const payload = {
    data: [
      false, // is_reset_val
      { path: uploadedPath, meta: { _type: "gradio.FileData" } },
      rotateDeg,
      moveForward,
      verticalTilt,
      wideangle,
      0, // seed
      true, // randomize_seed
      DEFAULT_GUIDANCE_SCALE,
      DEFAULT_INFERENCE_STEPS,
      DEFAULT_WIDTH,
      DEFAULT_HEIGHT,
      null, // prev_output
    ],
  };

  const response = await fetch(`${HF_SPACE_URL}/gradio_api/call/maybe_infer`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...authHeaders(token),
    },
    body: JSON.stringify(payload),
  });

  if (response.status === 429) {
    throw new Error("Rate limit / quota exceeded (429)");
  }

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Submit failed (${response.status}): ${text.slice(0, 300)}`);
  }

  const data = await response.json();
  const eventId = data.event_id;
  if (!eventId) throw new Error("No event_id in response");
  return eventId;
}

/** Poll for result via SSE endpoint, returns the image URL */
async function pollResult(eventId: string, token: string): Promise<string> {
  const response = await fetch(
    `${HF_SPACE_URL}/gradio_api/call/maybe_infer/${eventId}`,
    { headers: authHeaders(token) },
  );

  if (response.status === 429) {
    throw new Error("Rate limit / quota exceeded while fetching result (429)");
  }

  if (!response.ok) {
    throw new Error(`Result fetch failed (${response.status})`);
  }

  const text = await response.text();
  return parseSSEResponse(text);
}

function parseSSEResponse(text: string): string {
  const lines = text.split("\n");
  let errorMsg = "";
  let isErrorEvent = false;

  for (const line of lines) {
    if (line.startsWith("event: error")) {
      isErrorEvent = true;
      errorMsg = "API returned an error";
      continue;
    }
    if (line.startsWith("data: ")) {
      const dataStr = line.substring(6).trim();
      if (dataStr === "null") continue;

      if (isErrorEvent) {
        try {
          const errData = JSON.parse(dataStr);
          if (typeof errData === "string") errorMsg = errData;
          else if (typeof errData === "object" && errData !== null) {
            errorMsg = errData.message || JSON.stringify(errData);
          }
        } catch {
          errorMsg = dataStr;
        }
        throw new Error(errorMsg);
      }

      try {
        const data = JSON.parse(dataStr);
        if (Array.isArray(data) && data.length > 0) {
          const first = data[0];
          if (typeof first === "object" && first !== null && "url" in first) {
            return first.url;
          }
        }
      } catch {
        continue;
      }
    } else {
      isErrorEvent = false;
    }
  }

  throw new Error(errorMsg || "No result image found in SSE response");
}

/** Download image from URL and return as base64 data */
async function downloadImage(
  imageUrl: string,
  token: string,
): Promise<{ base64: string; contentType: string }> {
  const response = await fetch(imageUrl, {
    headers: authHeaders(token),
  });

  if (!response.ok) {
    throw new Error(`Image download failed (${response.status})`);
  }

  const blob = await response.blob();
  const contentType = blob.type || "image/webp";

  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      // Remove the data:...;base64, prefix
      const base64 = result.split(",")[1];
      resolve({ base64, contentType });
    };
    reader.onerror = () => reject(new Error("Failed to read image blob"));
    reader.readAsDataURL(blob);
  });
}

export interface GeneratedAngle {
  name: string;
  success: boolean;
  image_data?: string;
  content_type?: string;
  error?: string;
}

/**
 * Generate a single angle with retries and token rotation.
 * Each retry uses a different token to spread quota usage.
 */
async function generateSingleAngle(
  uploadedPath: string,
  rotateDeg: number,
  moveForward: number,
  verticalTilt: number,
  wideangle: boolean,
  preferredToken?: string,
): Promise<{ base64: string; contentType: string }> {
  let lastError = "";
  const startToken = preferredToken || getNextToken();

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    // Use a different token on each retry
    const token = attempt === 0 ? startToken : getNextToken();

    try {
      // Upload image is already done outside, we reuse the path
      const eventId = await submitJob(
        uploadedPath, rotateDeg, moveForward, verticalTilt, wideangle, token,
      );
      const imageUrl = await pollResult(eventId, token);
      return await downloadImage(imageUrl, token);
    } catch (e) {
      lastError = e instanceof Error ? e.message : String(e);
      const quota = isQuotaError(lastError);
      console.warn(
        `Generation attempt ${attempt + 1}/${MAX_RETRIES} failed${quota ? " (QUOTA)" : ""}: ${lastError}`,
      );

      if (attempt < MAX_RETRIES - 1) {
        const delay = quota
          ? QUOTA_RETRY_DELAY * (attempt + 1)
          : Math.min(RETRY_BASE_DELAY * Math.pow(2, attempt), 60000);
        await sleep(delay);
      }
    }
  }

  throw new Error(`Generation failed after ${MAX_RETRIES} attempts: ${lastError}`);
}

export interface ProgressCallback {
  (update: {
    type: "start" | "result" | "done";
    total?: number;
    completed?: number;
    result?: GeneratedAngle;
  }): void;
}

/**
 * Generate all 9 angles with deduplication and token rotation.
 * Calls onProgress for each completed angle.
 */
export async function generateAllAngles(
  imageFile: File,
  lens: string,
  onProgress: ProgressCallback,
): Promise<GeneratedAngle[]> {
  const total = PREDEFINED_ANGLES.length;
  onProgress({ type: "start", total });

  // Optimize image
  const optimizedBlob = await optimizeImage(imageFile);

  // Upload once with the first token
  const uploadToken = getNextToken();
  const uploadedPath = await uploadImage(optimizedBlob, uploadToken);

  const forward = convertForward(lens);
  const isWide = lens === "wide";

  // Deduplicate: angles that clamp to the same params share one result
  type ParamKey = string;
  const paramKeyFn = (r: number, f: number, t: number, w: boolean): ParamKey =>
    `${r}:${f}:${t}:${w}`;

  const uniqueParams = new Map<ParamKey, { rotate: number; tilt: number }>();
  const angleToKey = new Map<string, ParamKey>();

  for (const angle of PREDEFINED_ANGLES) {
    const rotate = clampRotate(angle.h);
    const tilt = convertVertical(angle.v);
    const key = paramKeyFn(rotate, forward, tilt, isWide);
    angleToKey.set(angle.name, key);
    if (!uniqueParams.has(key)) {
      uniqueParams.set(key, { rotate, tilt });
    }
  }

  console.log(
    `Deduplication: ${PREDEFINED_ANGLES.length} angles -> ${uniqueParams.size} unique API calls`,
  );

  // Generate unique params sequentially to minimize quota usage
  const paramResults = new Map<ParamKey, { base64: string; contentType: string } | Error>();
  let completed = 0;

  for (const [key, params] of uniqueParams) {
    const token = getNextToken();
    try {
      const result = await generateSingleAngle(
        uploadedPath, params.rotate, forward, params.tilt, isWide, token,
      );
      paramResults.set(key, result);
    } catch (e) {
      paramResults.set(key, e instanceof Error ? e : new Error(String(e)));
    }

    // Notify progress for all angles that share this param key
    for (const angle of PREDEFINED_ANGLES) {
      if (angleToKey.get(angle.name) === key) {
        completed++;
        const paramResult = paramResults.get(key);
        const genResult: GeneratedAngle =
          paramResult instanceof Error
            ? { name: angle.name, success: false, error: paramResult.message }
            : {
                name: angle.name,
                success: true,
                image_data: paramResult!.base64,
                content_type: paramResult!.contentType,
              };
        onProgress({ type: "result", completed, total, result: genResult });
      }
    }
  }

  // Collect all results
  const results: GeneratedAngle[] = PREDEFINED_ANGLES.map((angle) => {
    const key = angleToKey.get(angle.name)!;
    const paramResult = paramResults.get(key);
    if (paramResult instanceof Error) {
      return { name: angle.name, success: false, error: paramResult.message };
    }
    return {
      name: angle.name,
      success: true,
      image_data: paramResult!.base64,
      content_type: paramResult!.contentType,
    };
  });

  onProgress({ type: "done", completed: total, total });
  return results;
}

/**
 * Retry a single failed angle.
 */
export async function retrySingleAngle(
  imageFile: File,
  angleName: string,
  lens: string,
): Promise<GeneratedAngle> {
  const angleConfig = PREDEFINED_ANGLES.find((a) => a.name === angleName);
  if (!angleConfig) throw new Error(`Unknown angle: ${angleName}`);

  const optimizedBlob = await optimizeImage(imageFile);
  const token = getNextToken();
  const uploadedPath = await uploadImage(optimizedBlob, token);

  const forward = convertForward(lens);
  const rotate = clampRotate(angleConfig.h);
  const tilt = convertVertical(angleConfig.v);
  const isWide = lens === "wide";

  try {
    const result = await generateSingleAngle(
      uploadedPath, rotate, forward, tilt, isWide,
    );
    return {
      name: angleName,
      success: true,
      image_data: result.base64,
      content_type: result.contentType,
    };
  } catch (e) {
    return {
      name: angleName,
      success: false,
      error: e instanceof Error ? e.message : String(e),
    };
  }
}
