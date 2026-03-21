/**
 * HuggingFace Gradio Space client for direct browser-to-HF API calls.
 * Eliminates the need for a separate backend server.
 */

const HF_SPACE_URL = "https://linoyts-qwen-image-edit-angles.hf.space";

const DEFAULT_GUIDANCE_SCALE = 1.0;
const DEFAULT_INFERENCE_STEPS = 4;
const DEFAULT_WIDTH = 1024;
const DEFAULT_HEIGHT = 1024;
const MAX_RETRIES = 3;
const RETRY_BASE_DELAY = 2000; // ms

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
  if (90 < deg && deg <= 180) return 90.0;
  if (-180 <= deg && deg < -90) return -90.0;
  return 0.0;
}

function convertVertical(v: number): number {
  return Math.max(-1.0, Math.min(1.0, v / 60.0));
}

export function convertForward(lens: string): number {
  const mapping: Record<string, number> = { closeup: 5.0, wide: 0.0, normal: 2.0 };
  return mapping[lens] ?? 2.0;
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function uploadImage(imageData: Blob): Promise<string> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const formData = new FormData();
      formData.append("files", imageData, "input.png");

      const response = await fetch(`${HF_SPACE_URL}/gradio_api/upload`, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const result = await response.json();
        if (Array.isArray(result) && result.length > 0) {
          return result[0];
        }
        throw new Error(`Unexpected upload response: ${JSON.stringify(result)}`);
      }

      console.warn(`Upload attempt ${attempt + 1} failed: ${response.status}`);
    } catch (e) {
      if (e instanceof TypeError && e.message.includes("fetch")) {
        console.warn(`Upload attempt ${attempt + 1} network error`);
      } else if (attempt === MAX_RETRIES - 1) {
        throw e;
      }
    }

    if (attempt < MAX_RETRIES - 1) {
      const delay = Math.min(RETRY_BASE_DELAY * Math.pow(2, attempt), 30000);
      await sleep(delay);
    }
  }

  throw new Error("Failed to upload image after all retries");
}

function parseSSEResponse(text: string): string {
  const lines = text.split("\n");
  let errorMsg = "";

  for (const line of lines) {
    if (line.startsWith("event: error")) {
      errorMsg = "API returned an error";
    }
    if (line.startsWith("data: ")) {
      const dataStr = line.substring(6).trim();
      if (dataStr === "null") continue;
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
    }
  }

  throw new Error(errorMsg || "No result image found in SSE response");
}

async function generateAngle(
  uploadedPath: string,
  rotateDeg: number,
  moveForward: number,
  verticalTilt: number,
  wideangle: boolean,
  seed: number = 0,
  randomizeSeed: boolean = true,
): Promise<{ imageData: string; contentType: string }> {
  const payload = {
    data: [
      false, // is_reset_val
      { path: uploadedPath, meta: { _type: "gradio.FileData" } },
      rotateDeg,
      moveForward,
      verticalTilt,
      wideangle,
      seed,
      randomizeSeed,
      DEFAULT_GUIDANCE_SCALE,
      DEFAULT_INFERENCE_STEPS,
      DEFAULT_WIDTH,
      DEFAULT_HEIGHT,
      null, // prev_output
    ],
  };

  // Step 1: Submit the job
  const submitResponse = await fetch(`${HF_SPACE_URL}/gradio_api/call/maybe_infer`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!submitResponse.ok) {
    const text = await submitResponse.text();
    throw new Error(`Submit failed: ${submitResponse.status} - ${text.substring(0, 300)}`);
  }

  const submitResult = await submitResponse.json();
  const eventId = submitResult.event_id;
  if (!eventId) {
    throw new Error("No event_id in submit response");
  }

  // Step 2: Poll for result (SSE stream)
  const resultResponse = await fetch(
    `${HF_SPACE_URL}/gradio_api/call/maybe_infer/${eventId}`,
  );

  if (!resultResponse.ok) {
    throw new Error(`Result fetch failed: ${resultResponse.status}`);
  }

  const resultText = await resultResponse.text();
  const imageUrl = parseSSEResponse(resultText);

  // Step 3: Download the generated image
  const imageResponse = await fetch(imageUrl);

  if (!imageResponse.ok) {
    throw new Error(`Image download failed: ${imageResponse.status}`);
  }

  const contentType = imageResponse.headers.get("content-type") || "image/webp";
  const imageBlob = await imageResponse.blob();

  // Convert to base64
  const base64 = await blobToBase64(imageBlob);

  return { imageData: base64, contentType };
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      // Remove the data:...;base64, prefix
      const base64 = result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function generateAngleWithRetry(
  uploadedPath: string,
  rotateDeg: number,
  moveForward: number,
  verticalTilt: number,
  wideangle: boolean,
): Promise<{ imageData: string; contentType: string }> {
  let lastError: Error | null = null;

  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      return await generateAngle(uploadedPath, rotateDeg, moveForward, verticalTilt, wideangle);
    } catch (e) {
      lastError = e instanceof Error ? e : new Error(String(e));
      console.warn(`Generation attempt ${attempt + 1}/${MAX_RETRIES} failed: ${lastError.message}`);
      if (attempt < MAX_RETRIES - 1) {
        const delay = Math.min(RETRY_BASE_DELAY * Math.pow(2, attempt), 30000);
        await sleep(delay);
      }
    }
  }

  throw new Error(`Generation failed after ${MAX_RETRIES} attempts: ${lastError?.message}`);
}

async function optimizeImage(file: File): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(file);

    img.onload = () => {
      URL.revokeObjectURL(url);
      const maxSize = 2048;
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
      if (!ctx) {
        reject(new Error("Cannot get canvas context"));
        return;
      }
      ctx.drawImage(img, 0, 0, w, h);
      canvas.toBlob(
        (blob) => {
          if (blob) resolve(blob);
          else reject(new Error("Canvas toBlob failed"));
        },
        "image/png",
      );
    };

    img.onerror = () => {
      URL.revokeObjectURL(url);
      reject(new Error("Failed to load image"));
    };

    img.src = url;
  });
}

export interface AngleResult {
  name: string;
  success: boolean;
  image_data?: string;
  content_type?: string;
  error?: string;
}

export type OnProgress = (result: AngleResult, completed: number, total: number) => void;

export async function generateAllAngles(
  file: File,
  lens: string,
  onProgress?: OnProgress,
): Promise<AngleResult[]> {
  const optimized = await optimizeImage(file);
  const uploadedPath = await uploadImage(optimized);

  const forward = convertForward(lens);
  const total = PREDEFINED_ANGLES.length;
  const results: AngleResult[] = [];
  let completed = 0;

  // Process with concurrency limit of 3
  const concurrency = 3;
  const queue = [...PREDEFINED_ANGLES];

  const worker = async () => {
    while (queue.length > 0) {
      const angle = queue.shift();
      if (!angle) break;

      const rotate = clampRotate(angle.h);
      const tilt = convertVertical(angle.v);

      try {
        const result = await generateAngleWithRetry(
          uploadedPath,
          rotate,
          forward,
          tilt,
          lens === "wide",
        );

        const angleResult: AngleResult = {
          name: angle.name,
          success: true,
          image_data: result.imageData,
          content_type: result.contentType,
        };

        completed++;
        results.push(angleResult);
        onProgress?.(angleResult, completed, total);
      } catch (e) {
        const angleResult: AngleResult = {
          name: angle.name,
          success: false,
          error: e instanceof Error ? e.message : String(e),
        };

        completed++;
        results.push(angleResult);
        onProgress?.(angleResult, completed, total);
      }
    }
  };

  // Launch concurrent workers
  const workers = Array.from({ length: Math.min(concurrency, total) }, () => worker());
  await Promise.all(workers);

  return results;
}

export async function retrySingleAngle(
  file: File,
  angleName: string,
  lens: string,
): Promise<AngleResult> {
  const angleConfig = PREDEFINED_ANGLES.find((a) => a.name === angleName);
  if (!angleConfig) {
    return { name: angleName, success: false, error: `Unknown angle: ${angleName}` };
  }

  try {
    const optimized = await optimizeImage(file);
    const uploadedPath = await uploadImage(optimized);
    const forward = convertForward(lens);
    const rotate = clampRotate(angleConfig.h);
    const tilt = convertVertical(angleConfig.v);

    const result = await generateAngleWithRetry(
      uploadedPath,
      rotate,
      forward,
      tilt,
      lens === "wide",
    );

    return {
      name: angleName,
      success: true,
      image_data: result.imageData,
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
