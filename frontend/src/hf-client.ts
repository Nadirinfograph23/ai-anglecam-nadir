const HF_SPACE_URL = "https://linoyts-qwen-image-edit-angles.hf.space";

const HF_TOKENS: string[] = [
  import.meta.env.VITE_HF_TOKEN_1 || "",
  import.meta.env.VITE_HF_TOKEN_2 || "",
  import.meta.env.VITE_HF_TOKEN_3 || "",
  import.meta.env.VITE_HF_TOKEN_4 || "",
  import.meta.env.VITE_HF_TOKEN_5 || "",
  import.meta.env.VITE_HF_TOKEN_6 || "",
].filter((t) => t.length > 0);

const GENERATION_DEFAULTS = {
  guidanceScale: 1.0,
  inferenceSteps: 4,
  width: 1024,
  height: 1024,
};

const MAX_RETRIES = Math.max(4, HF_TOKENS.length * 2);
const RETRY_BASE_DELAY = 1000;
const RETRY_MAX_DELAY = 15000;
const MAX_CONCURRENT = Math.max(3, HF_TOKENS.length * 2);

class TokenRotator {
  private tokens: string[];
  private index = 0;
  private failedTokens: Map<string, number> = new Map();
  private cooldownMs = 60000;

  constructor(tokens: string[]) {
    this.tokens = tokens.length > 0 ? tokens : [""];
  }

  getNext(): string {
    const now = Date.now();
    for (let i = 0; i < this.tokens.length; i++) {
      const token = this.tokens[this.index];
      this.index = (this.index + 1) % this.tokens.length;
      const cooldownUntil = this.failedTokens.get(token) || 0;
      if (now >= cooldownUntil) {
        return token;
      }
    }
    // All on cooldown - return the one with earliest expiry
    let earliest = "";
    let earliestTime = Infinity;
    for (const [token, time] of this.failedTokens) {
      if (time < earliestTime) {
        earliestTime = time;
        earliest = token;
      }
    }
    return earliest || this.tokens[0];
  }

  markFailed(token: string): void {
    this.failedTokens.set(token, Date.now() + this.cooldownMs);
  }

  markSuccess(token: string): void {
    this.failedTokens.delete(token);
  }
}

const tokenRotator = new TokenRotator(HF_TOKENS);

interface AngleConfig {
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
  return Math.max(-1, Math.min(1, v / 60));
}

function convertForward(lens: string): number {
  const mapping: Record<string, number> = { closeup: 5, wide: 0, normal: 2 };
  return mapping[lens] ?? 2;
}

function getHeaders(contentType?: string): { headers: Record<string, string>; token: string } {
  const token = tokenRotator.getNext();
  const headers: Record<string, string> = {};
  if (contentType) headers["Content-Type"] = contentType;
  if (token) headers["Authorization"] = `Bearer ${token}`;
  return { headers, token };
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function uploadImage(file: File): Promise<string> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const { headers, token } = getHeaders();
    delete headers["Content-Type"];
    const formData = new FormData();
    formData.append("files", file, "input.png");

    try {
      const response = await fetch(`${HF_SPACE_URL}/gradio_api/upload`, {
        method: "POST",
        headers,
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        if (Array.isArray(result) && result.length > 0) {
          tokenRotator.markSuccess(token);
          return result[0];
        }
        throw new Error("Unexpected upload response");
      }

      if (response.status === 429) {
        tokenRotator.markFailed(token);
      }
    } catch {
      // retry
    }

    if (attempt < MAX_RETRIES - 1) {
      await sleep(Math.min(RETRY_BASE_DELAY * 2 ** attempt, RETRY_MAX_DELAY));
    }
  }
  throw new Error("Failed to upload image after all retries");
}

async function generateAngle(
  uploadedPath: string,
  rotateDeg: number,
  moveForward: number,
  verticalTilt: number,
  wideangle: boolean,
): Promise<{ imageData: string; contentType: string }> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    const { headers, token } = getHeaders("application/json");

    try {
      const payload = {
        data: [
          false,
          { path: uploadedPath, meta: { _type: "gradio.FileData" } },
          rotateDeg,
          moveForward,
          verticalTilt,
          wideangle,
          0,
          true,
          GENERATION_DEFAULTS.guidanceScale,
          GENERATION_DEFAULTS.inferenceSteps,
          GENERATION_DEFAULTS.width,
          GENERATION_DEFAULTS.height,
          null,
        ],
      };

      const submitResponse = await fetch(
        `${HF_SPACE_URL}/gradio_api/call/maybe_infer`,
        { method: "POST", headers, body: JSON.stringify(payload) },
      );

      if (submitResponse.status === 429) {
        tokenRotator.markFailed(token);
        throw new Error("Rate limited");
      }

      if (!submitResponse.ok) {
        throw new Error(`Submit failed: ${submitResponse.status}`);
      }

      const submitData = await submitResponse.json();
      const eventId = submitData.event_id;
      if (!eventId) throw new Error("No event_id in response");

      const resultHeaders: Record<string, string> = {};
      if (token) resultHeaders["Authorization"] = `Bearer ${token}`;

      const resultResponse = await fetch(
        `${HF_SPACE_URL}/gradio_api/call/maybe_infer/${eventId}`,
        { headers: resultHeaders },
      );

      if (resultResponse.status === 429) {
        tokenRotator.markFailed(token);
        throw new Error("Rate limited during result fetch");
      }

      if (!resultResponse.ok) {
        throw new Error(`Result fetch failed: ${resultResponse.status}`);
      }

      const text = await resultResponse.text();
      const imageUrl = parseSSEResponse(text);

      const imageResponse = await fetch(imageUrl, { headers: resultHeaders });
      if (!imageResponse.ok) {
        throw new Error(`Image download failed: ${imageResponse.status}`);
      }

      tokenRotator.markSuccess(token);
      const blob = await imageResponse.blob();
      const contentType = blob.type || "image/webp";
      const arrayBuf = await blob.arrayBuffer();
      const base64 = btoa(
        new Uint8Array(arrayBuf).reduce((data, byte) => data + String.fromCharCode(byte), ""),
      );

      return { imageData: base64, contentType };
    } catch (e) {
      if (attempt < MAX_RETRIES - 1) {
        await sleep(Math.min(RETRY_BASE_DELAY * 2 ** attempt, RETRY_MAX_DELAY));
      } else {
        throw e;
      }
    }
  }
  throw new Error("Generation failed after all retries");
}

function parseSSEResponse(text: string): string {
  const lines = text.split("\n");
  for (const line of lines) {
    if (line.startsWith("data: ")) {
      const dataStr = line.substring(6).trim();
      if (dataStr === "null") continue;
      try {
        const data = JSON.parse(dataStr);
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
  throw new Error("No result image found in SSE response");
}

export interface AngleResult {
  name: string;
  success: boolean;
  image_data?: string;
  content_type?: string;
  error?: string;
}

export type OnProgress = (result: AngleResult, completed: number, total: number) => void;

async function runWithConcurrency<T>(
  tasks: (() => Promise<T>)[],
  maxConcurrent: number,
): Promise<T[]> {
  const results: T[] = new Array(tasks.length);
  let nextIndex = 0;

  async function runNext(): Promise<void> {
    while (nextIndex < tasks.length) {
      const idx = nextIndex++;
      results[idx] = await tasks[idx]();
    }
  }

  const workers = Array.from(
    { length: Math.min(maxConcurrent, tasks.length) },
    () => runNext(),
  );
  await Promise.all(workers);
  return results;
}

export async function generateAllAngles(
  file: File,
  lens: string,
  onProgress: OnProgress,
): Promise<AngleResult[]> {
  const uploadedPath = await uploadImage(file);
  const forward = convertForward(lens);
  const total = PREDEFINED_ANGLES.length;
  let completed = 0;

  const tasks = PREDEFINED_ANGLES.map((angle) => async (): Promise<AngleResult> => {
    const rotate = clampRotate(angle.h);
    const tilt = convertVertical(angle.v);
    try {
      const { imageData, contentType } = await generateAngle(
        uploadedPath,
        rotate,
        forward,
        tilt,
        lens === "wide",
      );
      completed++;
      const result: AngleResult = {
        name: angle.name,
        success: true,
        image_data: imageData,
        content_type: contentType,
      };
      onProgress(result, completed, total);
      return result;
    } catch (e) {
      completed++;
      const result: AngleResult = {
        name: angle.name,
        success: false,
        error: e instanceof Error ? e.message : "Unknown error",
      };
      onProgress(result, completed, total);
      return result;
    }
  });

  return runWithConcurrency(tasks, MAX_CONCURRENT);
}

export async function retrySingleAngle(
  file: File,
  angleName: string,
  lens: string,
): Promise<AngleResult> {
  const angleConfig = PREDEFINED_ANGLES.find((a) => a.name === angleName);
  if (!angleConfig) throw new Error(`Unknown angle: ${angleName}`);

  const uploadedPath = await uploadImage(file);
  const forward = convertForward(lens);
  const rotate = clampRotate(angleConfig.h);
  const tilt = convertVertical(angleConfig.v);

  try {
    const { imageData, contentType } = await generateAngle(
      uploadedPath,
      rotate,
      forward,
      tilt,
      lens === "wide",
    );
    return {
      name: angleName,
      success: true,
      image_data: imageData,
      content_type: contentType,
    };
  } catch (e) {
    return {
      name: angleName,
      success: false,
      error: e instanceof Error ? e.message : "Unknown error",
    };
  }
}
