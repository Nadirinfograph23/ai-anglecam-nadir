/**
 * Gradio API client using the official @gradio/client library.
 *
 * Uses the /infer_edit_camera_angles endpoint for reliable image generation.
 * The official client handles connection management, file uploads, and
 * SSE parsing automatically.
 */

import { Client } from "@gradio/client";

const PRIMARY_SPACE = "linoyts/qwen-image-edit-angles";

// API-valid ranges: rotate [-90, 90], forward [0, 10], vertical [-1, 1]
// 9 distinct angles within valid parameter ranges
export const PREDEFINED_ANGLES = [
  { name: "Front",       h:   0, v:  0.0,  forward: 2.0 },
  { name: "Front Right", h:  45, v:  0.15, forward: 2.0 },
  { name: "Right",       h:  90, v:  0.0,  forward: 2.0 },
  { name: "Back Right",  h:  90, v:  0.5,  forward: 4.0 },
  { name: "Back",        h:   0, v: -0.7,  forward: 6.0 },
  { name: "Back Left",   h: -90, v:  0.5,  forward: 4.0 },
  { name: "Left",        h: -90, v:  0.0,  forward: 2.0 },
  { name: "Front Left",  h: -45, v:  0.15, forward: 2.0 },
  { name: "Top View",    h:   0, v:  0.8,  forward: 1.0 },
];

// Generation defaults
const DEFAULT_GUIDANCE_SCALE = 1.0;
const DEFAULT_INFERENCE_STEPS = 4;

// Retry configuration
const MAX_RETRIES = 3;
const RETRY_BASE_DELAY = 3000; // ms

function convertForward(lens: string): number {
  const mapping: Record<string, number> = { closeup: 5.0, wide: 0.0, normal: 2.0 };
  return mapping[lens] ?? 2.0;
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Resize and optimize image before upload to reduce bandwidth.
 */
async function optimizeImage(file: File, maxSize = 1536): Promise<Blob> {
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
        "image/jpeg",
        0.85
      );
    };
    img.onerror = () => reject(new Error("Failed to load image"));
    img.src = URL.createObjectURL(file);
  });
}

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = reader.result as string;
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
 * Generate a single angle using the official Gradio client.
 */
async function generateSingleAngle(
  client: InstanceType<typeof Client>,
  imageBlob: Blob,
  rotateDeg: number,
  moveForward: number,
  verticalTilt: number,
  wideangle: boolean
): Promise<{ imageData: string; contentType: string }> {
  for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
    try {
      const result = await client.predict("/infer_edit_camera_angles", {
        image: imageBlob,
        rotate_deg: rotateDeg,
        move_forward: moveForward,
        vertical_tilt: verticalTilt,
        wideangle: wideangle,
        seed: 0,
        randomize_seed: true,
        true_guidance_scale: DEFAULT_GUIDANCE_SCALE,
        num_inference_steps: DEFAULT_INFERENCE_STEPS,
        height: null,
        width: null,
        prev_output: null,
      });

      // The result.data contains [image_data, seed, prompt]
      const data = result.data as Array<unknown>;
      if (!data || data.length === 0) {
        throw new Error("Empty response from API");
      }

      const imageResult = data[0] as { url?: string; path?: string } | null;
      if (!imageResult) {
        throw new Error("No image in response");
      }

      // Get the image URL from the result
      const imageUrl = imageResult.url || imageResult.path;
      if (!imageUrl) {
        throw new Error("No image URL in response");
      }

      // Download the generated image
      const imageResponse = await fetch(imageUrl);
      if (!imageResponse.ok) {
        throw new Error("Image download failed: " + imageResponse.status);
      }

      const resultBlob = await imageResponse.blob();
      const contentType = imageResponse.headers.get("content-type") || "image/webp";
      const base64 = await blobToBase64(resultBlob);

      return { imageData: base64, contentType };
    } catch (e) {
      const errMsg = e instanceof Error ? e.message : String(e);
      console.warn("Angle generation attempt " + (attempt + 1) + " failed: " + errMsg);

      if (attempt < MAX_RETRIES - 1) {
        const delay = RETRY_BASE_DELAY * Math.pow(2, attempt) + Math.random() * 2000;
        await sleep(delay);
        continue;
      }
      throw e;
    }
  }
  throw new Error("Generation failed after all retries");
}

/**
 * Generate all 9 angles with streaming progress updates.
 * Processes angles with controlled concurrency to avoid overwhelming the API.
 */
export async function generateAllAnglesStream(
  imageFile: File,
  lens: string,
  callbacks: StreamCallbacks
): Promise<void> {
  const total = PREDEFINED_ANGLES.length;
  callbacks.onStart(total);

  try {
    // Connect to the Gradio Space
    const client = await Client.connect(PRIMARY_SPACE);

    // Optimize image once
    const optimized = await optimizeImage(imageFile);

    const defaultForward = convertForward(lens);
    let completed = 0;

    // Process angles with controlled concurrency (2 at a time)
    const concurrency = 2;
    const queue = [...PREDEFINED_ANGLES];
    const running: Promise<void>[] = [];

    const processAngle = async (
      angle: (typeof PREDEFINED_ANGLES)[number]
    ): Promise<void> => {
      const rotateDeg = angle.h;
      const verticalTilt = angle.v;
      const moveForward = angle.forward ?? defaultForward;
      const wideangle = lens === "wide";

      try {
        const result = await generateSingleAngle(
          client,
          optimized,
          rotateDeg,
          moveForward,
          verticalTilt,
          wideangle
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
  const angle = PREDEFINED_ANGLES.find((a) => a.name === angleName);
  if (!angle) {
    return { name: angleName, success: false, error: "Unknown angle: " + angleName };
  }

  try {
    const client = await Client.connect(PRIMARY_SPACE);
    const optimized = await optimizeImage(imageFile);

    const defaultForward = convertForward(lens);
    const rotateDeg = angle.h;
    const verticalTilt = angle.v;
    const moveForward = angle.forward ?? defaultForward;
    const wideangle = lens === "wide";

    const result = await generateSingleAngle(
      client,
      optimized,
      rotateDeg,
      moveForward,
      verticalTilt,
      wideangle
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
