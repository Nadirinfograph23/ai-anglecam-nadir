/**
 * Gradio API client with dual-mode: Vercel API proxy (primary) + direct HF fallback.
 *
 * Primary mode: Calls /api/generate on Vercel (server-side token rotation, unlimited quota).
 * Fallback mode: Direct @gradio/client connection to HF Space (when API unavailable).
 *
 * Features:
 * - Request queue with controlled concurrency
 * - Client-side result caching
 * - Automatic retry with exponential backoff
 * - Graceful fallback between API modes
 * - Pressure handling via sequential processing
 */

import { Client } from "@gradio/client";

const PRIMARY_SPACE = "linoyts/qwen-image-edit-angles";

// Generation defaults (used for direct HF fallback)
const DEFAULT_GUIDANCE_SCALE = 1.0;
const DEFAULT_INFERENCE_STEPS = 4;

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

// Retry configuration
const MAX_RETRIES = 3;
const RETRY_BASE_DELAY = 3000; // ms
// Concurrency: process 1 angle at a time for maximum reliability under pressure
const CONCURRENCY = 1;

// Client-side cache for generated images (survives within session)
const imageCache = new Map<string, { imageData: string; contentType: string }>();

// Track whether the Vercel API proxy is available
let apiProxyAvailable: boolean | null = null; // null = not yet tested

function makeCacheKey(
  imageHash: string,
  rotate: number,
  forward: number,
  tilt: number,
  wide: boolean
): string {
  return `${imageHash}:${rotate}:${forward}:${tilt}:${wide}`;
}

function convertForward(lens: string): number {
  const mapping: Record<string, number> = { closeup: 5.0, wide: 0.0, normal: 2.0 };
  return mapping[lens] ?? 2.0;
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Compute a simple hash of the image file for cache keys.
 */
async function hashFile(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest("SHA-256", buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray
    .slice(0, 8)
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("");
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

// --- Mode 1: Vercel API Proxy (server-side token rotation, unlimited quota) ---

/**
 * Generate a single angle via the Vercel API proxy.
 * This keeps HF tokens server-side for unlimited quota.
 */
async function generateViaAPI(
  imageBase64: string,
  angleName: string,
  rotateDeg: number,
  moveForward: number,
  verticalTilt: number,
  wideangle: boolean
): Promise<{ imageData: string; contentType: string }> {
  const response = await fetch("/api/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      image_base64: imageBase64,
      angle_name: angleName,
      rotate_deg: rotateDeg,
      move_forward: moveForward,
      vertical_tilt: verticalTilt,
      wideangle,
      seed: 0,
      randomize_seed: true,
    }),
  });

  if (!response.ok) {
    const errData = await response.json().catch(() => null);
    throw new Error(errData?.error || "API error: " + response.status);
  }

  const data = await response.json();
  if (!data.success) {
    throw new Error(data.error || "Generation failed");
  }

  return { imageData: data.image_data, contentType: data.content_type };
}

// --- Mode 2: Direct HF Space via @gradio/client (fallback) ---

/**
 * Generate a single angle using the official Gradio client directly.
 * Used as fallback when Vercel API proxy is unavailable.
 */
async function generateViaGradio(
  client: InstanceType<typeof Client>,
  imageBlob: Blob,
  rotateDeg: number,
  moveForward: number,
  verticalTilt: number,
  wideangle: boolean
): Promise<{ imageData: string; contentType: string }> {
  const result = await client.predict("/infer_edit_camera_angles", {
    image: imageBlob,
    rotate_deg: rotateDeg,
    move_forward: moveForward,
    vertical_tilt: verticalTilt,
    wideangle,
    seed: 0,
    randomize_seed: true,
    true_guidance_scale: DEFAULT_GUIDANCE_SCALE,
    num_inference_steps: DEFAULT_INFERENCE_STEPS,
    height: null,
    width: null,
    prev_output: null,
  });

  const data = result.data as Array<unknown>;
  if (!data || data.length === 0) {
    throw new Error("Empty response from API");
  }

  const imageResult = data[0] as { url?: string; path?: string } | null;
  if (!imageResult) {
    throw new Error("No image in response");
  }

  const imageUrl = imageResult.url || imageResult.path;
  if (!imageUrl) {
    throw new Error("No image URL in response");
  }

  const imageResponse = await fetch(imageUrl);
  if (!imageResponse.ok) {
    throw new Error("Image download failed: " + imageResponse.status);
  }

  const resultBlob = await imageResponse.blob();
  const contentType = imageResponse.headers.get("content-type") || "image/webp";
  const base64 = await blobToBase64(resultBlob);

  return { imageData: base64, contentType };
}

// --- Unified generation with auto-fallback ---

/**
 * Generate a single angle with retry logic and automatic fallback.
 * 1. Try Vercel API proxy first (server-side token rotation = unlimited quota)
 * 2. If proxy unavailable, fall back to direct @gradio/client
 */
async function generateSingleAngle(
  imageBase64: string,
  imageBlob: Blob,
  angleName: string,
  rotateDeg: number,
  moveForward: number,
  verticalTilt: number,
  wideangle: boolean,
  gradioClient: InstanceType<typeof Client> | null
): Promise<{ imageData: string; contentType: string }> {
  // Try API proxy first (if not already known to be unavailable)
  if (apiProxyAvailable !== false) {
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        const result = await generateViaAPI(
          imageBase64,
          angleName,
          rotateDeg,
          moveForward,
          verticalTilt,
          wideangle
        );
        apiProxyAvailable = true;
        return result;
      } catch (e) {
        const errMsg = e instanceof Error ? e.message : String(e);

        // If API route itself is missing (404), switch to direct mode
        if (errMsg.includes("404") || errMsg.includes("Not Found")) {
          console.warn("[gradio-client] API proxy not available, switching to direct HF mode");
          apiProxyAvailable = false;
          break;
        }

        console.warn(
          "[API] " + angleName + " attempt " + (attempt + 1) + " failed: " + errMsg
        );

        if (attempt < MAX_RETRIES - 1) {
          const delay = RETRY_BASE_DELAY * Math.pow(2, attempt) + Math.random() * 2000;
          await sleep(delay);
        }
      }
    }

    // If API proxy was available but all retries failed, still try direct
    if (apiProxyAvailable === true) {
      console.warn("[gradio-client] API proxy retries exhausted, trying direct HF");
    }
  }

  // Fallback: direct Gradio client
  if (gradioClient) {
    for (let attempt = 0; attempt < MAX_RETRIES; attempt++) {
      try {
        return await generateViaGradio(
          gradioClient,
          imageBlob,
          rotateDeg,
          moveForward,
          verticalTilt,
          wideangle
        );
      } catch (e) {
        const errMsg = e instanceof Error ? e.message : String(e);
        console.warn(
          "[HF] " + angleName + " attempt " + (attempt + 1) + " failed: " + errMsg
        );

        if (attempt < MAX_RETRIES - 1) {
          const delay = RETRY_BASE_DELAY * Math.pow(2, attempt) + Math.random() * 2000;
          await sleep(delay);
        }
      }
    }
  }

  throw new Error("Generation failed after all retries (API + direct HF)");
}

// --- Public API ---

/**
 * Generate all 9 angles with streaming progress updates.
 * Uses API proxy for unlimited quota, falls back to direct HF.
 * Processes angles with controlled concurrency for reliability under pressure.
 * Client-side caching avoids regenerating successful angles.
 */
export async function generateAllAnglesStream(
  imageFile: File,
  lens: string,
  callbacks: StreamCallbacks
): Promise<void> {
  const total = PREDEFINED_ANGLES.length;
  callbacks.onStart(total);

  try {
    // Optimize and convert image
    const optimized = await optimizeImage(imageFile);
    const imageBase64 = await blobToBase64(optimized);
    const imageHash = await hashFile(imageFile);

    // Try to connect Gradio client for fallback (non-blocking)
    let gradioClient: InstanceType<typeof Client> | null = null;
    try {
      gradioClient = await Client.connect(PRIMARY_SPACE);
    } catch (e) {
      console.warn("[gradio-client] Could not connect to HF Space for fallback:", e);
    }

    const defaultForward = convertForward(lens);
    let completed = 0;

    const queue = [...PREDEFINED_ANGLES];
    const running: Promise<void>[] = [];

    const processAngle = async (
      angle: (typeof PREDEFINED_ANGLES)[number]
    ): Promise<void> => {
      const rotateDeg = angle.h;
      const verticalTilt = angle.v;
      const moveForward = angle.forward ?? defaultForward;
      const wideangle = lens === "wide";

      // Check client-side cache
      const cacheKey = makeCacheKey(imageHash, rotateDeg, moveForward, verticalTilt, wideangle);
      const cached = imageCache.get(cacheKey);
      if (cached) {
        completed++;
        callbacks.onResult(
          {
            name: angle.name,
            success: true,
            image_data: cached.imageData,
            content_type: cached.contentType,
          },
          completed,
          total
        );
        return;
      }

      try {
        const result = await generateSingleAngle(
          imageBase64,
          optimized,
          angle.name,
          rotateDeg,
          moveForward,
          verticalTilt,
          wideangle,
          gradioClient
        );

        // Cache the result
        imageCache.set(cacheKey, result);

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
      while (running.length < CONCURRENCY && i < queue.length) {
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
    callbacks.onError(e instanceof Error ? e.message : "Generation failed");
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
    const optimized = await optimizeImage(imageFile);
    const imageBase64 = await blobToBase64(optimized);
    const imageHash = await hashFile(imageFile);

    const defaultForward = convertForward(lens);
    const rotateDeg = angle.h;
    const verticalTilt = angle.v;
    const moveForward = angle.forward ?? defaultForward;
    const wideangle = lens === "wide";

    // Try to connect Gradio client for fallback
    let gradioClient: InstanceType<typeof Client> | null = null;
    try {
      gradioClient = await Client.connect(PRIMARY_SPACE);
    } catch {
      // Fallback not available
    }

    const result = await generateSingleAngle(
      imageBase64,
      optimized,
      angle.name,
      rotateDeg,
      moveForward,
      verticalTilt,
      wideangle,
      gradioClient
    );

    // Cache the result
    const cacheKey = makeCacheKey(imageHash, rotateDeg, moveForward, verticalTilt, wideangle);
    imageCache.set(cacheKey, result);

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
