/**
 * Direct Gradio API client for HuggingFace Spaces.
 * 
 * Calls the Gradio Space API directly from the browser, eliminating
 * the need for a backend server. This approach:
 * - Distributes rate limits per-user (each browser has its own quota)
 * - Eliminates backend hosting costs
 * - Removes single point of failure
 */

const PRIMARY_SPACE = "https://linoyts-qwen-image-edit-angles.hf.space";

// Predefined camera angles with distinct h, v, and forward values
// Each angle produces a visually different result
export const PREDEFINED_ANGLES = [
  { name: "Front", h: 0, v: 0, forward: 2.0 },
  { name: "Front Right", h: 55, v: 10, forward: 2.0 },
  { name: "Right", h: 90, v: 0, forward: 2.0 },
  { name: "Back Right", h: 90, v: 40, forward: 3.5 },
  { name: "Back", h: 0, v: -50, forward: 4.0 },
  { name: "Back Left", h: -90, v: 40, forward: 3.5 },
  { name: "Left", h: -90, v: 0, forward: 2.0 },
  { name: "Front Left", h: -55, v: 10, forward: 2.0 },
  { name: "Top View", h: 0, v: 60, forward: 1.0 },
];

// Generation defaults
const DEFAULT_GUIDANCE_SCALE = 1.0;
const DEFAULT_INFERENCE_STEPS = 4;
const DEFAULT_WIDTH = 1024;
const DEFAULT_HEIGHT = 1024;

// Retry configuration
const MAX_RETRIES = 3;
const RETRY_BASE_DELAY = 2000; // ms

function convertVertical(v: number): number {
  return Math.max(-1.0, Math.min(1.0, v / 60.0));
}

function convertForward(lens: string): number {
  const mapping: Record<string, number> = { closeup: 5.0, wide: 0.0, normal: 2.0 };
  return mapping[lens] ?? 2.0;
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Resize and optimize image before upload to reduce bandwidth and improve speed.
 */
async function optimizeImage(file: File, maxSize = 2048): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      let { width, height } = img;
      if (Math.max(width, height) > maxSize) {
        const ratio = maxSize / Math.max(width, height);
        width = Math.round(width * ratio);
        height = Math.round(height * ratio);
      }
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        reject(new Error("Canvas not supported"));
        return;
      }
      ctx.drawImage(img, 0, 0, width, height);
      canvas.toBlob(
        (blob) => {
          if (blob) resolve(blob);
          else reject(new Error("Failed to create blob"));
        },
        "image/png",
        0.9
      );
    };
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = URL.createObjectURL(file);
  });
}

/**
 * Upload an image to the Gradio Space and get the file path.
 */
async function uploadImage(imageBlob: Blob, spaceUrl: string): Promise<string> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const formData = new FormData();
      formData.append("files", imageBlob, "input.png");

      const response = await fetch(`${spaceUrl}/gradio_api/upload`, {
        method: "POST",
        body: formData,
      });

      if (response.status === 429) {
        throw new Error("Rate limited (429)");
      }

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      const result = await response.json();
      if (Array.isArray(result) && result.length > 0) {
        return result[0];
      }
      throw new Error("Unexpected upload response");
    } catch (e) {
      if (attempt < MAX_RETRIES - 1) {
        await sleep(RETRY_BASE_DELAY * Math.pow(2, attempt));
        continue;
      }
      throw e;
    }
  }
  throw new Error("Upload failed after all retries");
}

/**
 * Parse SSE response to extract image URL.
 */
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
        if (data && typeof data === "object" && "error" in data) {
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

  throw new Error(errorMsg || "No result image found in response");
}

/**
 * Generate a single angle image via the Gradio API.
 */
async function generateSingleAngle(
  uploadedPath: string,
  rotateDeg: number,
  moveForward: number,
  verticalTilt: number,
  wideangle: boolean,
  spaceUrl: string
): Promise<{ imageData: string; contentType: string }> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      // Step 1: Submit the job
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

      const submitResponse = await fetch(
        `${spaceUrl}/gradio_api/call/maybe_infer`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );

      if (submitResponse.status === 429) {
        throw new Error("Rate limited (429)");
      }

      if (!submitResponse.ok) {
        const errText = await submitResponse.text().catch(() => "");
        throw new Error(
          `Submit failed: ${submitResponse.status} - ${errText.slice(0, 200)}`
        );
      }

      const submitData = await submitResponse.json();
      const eventId = submitData.event_id;
      if (!eventId) {
        throw new Error("No event_id in submit response");
      }

      // Step 2: Poll for result (SSE stream)
      const resultResponse = await fetch(
        `${spaceUrl}/gradio_api/call/maybe_infer/${eventId}`
      );

      if (resultResponse.status === 429) {
        throw new Error("Rate limited polling (429)");
      }

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

      const imageBlob = await imageResponse.blob();
      const contentType =
        imageResponse.headers.get("content-type") || "image/webp";

      // Convert to base64
      const base64 = await blobToBase64(imageBlob);

      return { imageData: base64, contentType };
    } catch (e) {
      if (attempt < MAX_RETRIES - 1) {
        const delay =
          RETRY_BASE_DELAY * Math.pow(1.5, attempt) +
          Math.random() * 1000;
        await sleep(delay);
        continue;
      }
      throw e;
    }
  }
  throw new Error("Generation failed after all retries");
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
      // Remove data URL prefix to get raw base64
      const base64 = result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

export interface AngleResult {
  name: string;
  success: boolean;
  image_data?: string;
  content_type?: string;
  error?: string;
}

export interface StreamCallbacks {
  onStart: (total: number) => void;
  onResult: (result: AngleResult, completed: number, total: number) => void;
  onDone: (completed: number, total: number) => void;
  onError: (message: string) => void;
}

/**
 * Generate all 9 angles with streaming progress updates.
 * Each angle is processed and results are reported as they complete.
 */
export async function generateAllAnglesStream(
  imageFile: File,
  lens: string,
  callbacks: StreamCallbacks
): Promise<void> {
  const spaceUrl = PRIMARY_SPACE;
  const total = PREDEFINED_ANGLES.length;

  callbacks.onStart(total);

  try {
    // Optimize and upload image once
    const optimized = await optimizeImage(imageFile);
    const uploadedPath = await uploadImage(optimized, spaceUrl);

    const defaultForward = convertForward(lens);
    let completed = 0;

    // Process angles with controlled concurrency (3 at a time)
    const concurrency = 3;
    const queue = [...PREDEFINED_ANGLES];
    const running: Promise<void>[] = [];

    const processAngle = async (
      angle: (typeof PREDEFINED_ANGLES)[number]
    ): Promise<void> => {
      const rotateDeg = angle.h;
      const verticalTilt = convertVertical(angle.v);
      const moveForward =
        angle.forward !== null && angle.forward !== undefined
          ? angle.forward
          : defaultForward;
      const wideangle = lens === "wide";

      try {
        const result = await generateSingleAngle(
          uploadedPath,
          rotateDeg,
          moveForward,
          verticalTilt,
          wideangle,
          spaceUrl
        );
        completed++;
        callbacks.onResult(
          {
            name: angle.name,
            success: true,
            image_data: result.imageData,
            content_type: result.contentType,
          },
          completed,
          total
        );
      } catch (e) {
        completed++;
        callbacks.onResult(
          {
            name: angle.name,
            success: false,
            error: e instanceof Error ? e.message : "Generation failed",
          },
          completed,
          total
        );
      }
    };

    // Process with concurrency control
    let i = 0;
    while (i < queue.length || running.length > 0) {
      while (running.length < concurrency && i < queue.length) {
        const angle = queue[i++];
        const promise = processAngle(angle).then(() => {
          const idx = running.indexOf(promise);
          if (idx !== -1) running.splice(idx, 1);
        });
        running.push(promise);
      }
      if (running.length > 0) {
        await Promise.race(running);
      }
    }

    callbacks.onDone(completed, total);
  } catch (e) {
    callbacks.onError(
      e instanceof Error ? e.message : "Generation failed"
    );
  }
}

/**
 * Retry generating a specific angle.
 */
export async function retryAngle(
  imageFile: File,
  angleName: string,
  lens: string
): Promise<AngleResult> {
  const spaceUrl = PRIMARY_SPACE;
  const angle = PREDEFINED_ANGLES.find((a) => a.name === angleName);
  if (!angle) {
    return { name: angleName, success: false, error: `Unknown angle: ${angleName}` };
  }

  try {
    const optimized = await optimizeImage(imageFile);
    const uploadedPath = await uploadImage(optimized, spaceUrl);

    const defaultForward = convertForward(lens);
    const rotateDeg = angle.h;
    const verticalTilt = convertVertical(angle.v);
    const moveForward =
      angle.forward !== null && angle.forward !== undefined
        ? angle.forward
        : defaultForward;
    const wideangle = lens === "wide";

    const result = await generateSingleAngle(
      uploadedPath,
      rotateDeg,
      moveForward,
      verticalTilt,
      wideangle,
      spaceUrl
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
      error: e instanceof Error ? e.message : "Retry failed",
    };
  }
}
