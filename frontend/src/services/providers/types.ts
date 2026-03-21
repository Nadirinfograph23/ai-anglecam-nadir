/**
 * Shared types for the multi-provider image generation system.
 */

export interface GenerationResult {
  imageData: string; // base64
  contentType: string;
  provider: string;
}

export interface AngleParams {
  rotateDeg: number;
  moveForward: number;
  verticalTilt: number;
  wideangle: boolean;
  angleName: string;
}

export interface ImageProvider {
  name: string;
  isAvailable(): boolean;
  generateAngle(
    imageBlob: Blob,
    params: AngleParams,
    timeoutMs: number,
  ): Promise<GenerationResult>;
}

/** Text prompts for each angle, used by fallback providers that rely on text guidance. */
export const ANGLE_PROMPTS: Record<string, string> = {
  Front: "front view, facing the camera directly, product photography",
  "Front Right": "front-right angle view, 45 degrees rotated to the right, product photography",
  Right: "right side view, 90 degrees from the front, product photography",
  "Back Right": "back-right angle view, 135 degrees rotated, product photography",
  Back: "back view, rear facing, product photography",
  "Back Left": "back-left angle view, rotated 225 degrees, product photography",
  Left: "left side view, 270 degrees from the front, product photography",
  "Front Left": "front-left angle view, 315 degrees rotated, product photography",
  "Top View": "top-down view, bird's eye perspective, overhead shot, product photography",
};

export function blobToBase64(blob: Blob): Promise<string> {
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

export function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms);
    promise.then(
      (val) => { clearTimeout(timer); resolve(val); },
      (err) => { clearTimeout(timer); reject(err); },
    );
  });
}
