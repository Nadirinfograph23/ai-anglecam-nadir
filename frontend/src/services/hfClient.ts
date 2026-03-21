/**
 * Image generation client using HuggingFace Qwen Image Edit.
 * Includes retry logic and caching for reliability.
 */

import { generateWithFallback } from "./fallbackController";
import type { AngleParams } from "./providers/types";
import { imageCache } from "./imageCache";

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

export function optimizeImage(file: File): Promise<Blob> {
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

export type OnProgress = (
  result: AngleResult,
  completed: number,
  total: number,
) => void;

export async function generateAllAngles(
  file: File,
  lens: string,
  onProgress?: OnProgress,
): Promise<AngleResult[]> {
  const optimized = await optimizeImage(file);
  const imageHash = await imageCache.computeImageHash(file);
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

      const params: AngleParams = {
        rotateDeg: rotate,
        moveForward: forward,
        verticalTilt: tilt,
        wideangle: lens === "wide",
        angleName: angle.name,
      };

      try {
        const result = await generateWithFallback(
          optimized,
          params,
          imageHash,
          lens,
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
    const imageHash = await imageCache.computeImageHash(file);
    const forward = convertForward(lens);
    const rotate = clampRotate(angleConfig.h);
    const tilt = convertVertical(angleConfig.v);

    const params: AngleParams = {
      rotateDeg: rotate,
      moveForward: forward,
      verticalTilt: tilt,
      wideangle: lens === "wide",
      angleName,
    };

    const result = await generateWithFallback(optimized, params, imageHash, lens);

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
