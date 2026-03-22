/**
 * Puter.js AI Image Generation Service
 *
 * Uses puter.ai.txt2img() with input_image for angle generation.
 * Falls back to the backend API if Puter.js is unavailable or fails.
 */

declare const puter: {
  ai: {
    txt2img: (
      prompt: string | { prompt: string; [key: string]: unknown },
      options?: Record<string, unknown>
    ) => Promise<HTMLImageElement>;
  };
};

export interface AngleConfig {
  name: string;
  prompt: string;
}

/** Descriptive prompts for each camera angle transformation */
export const ANGLE_PROMPTS: AngleConfig[] = [
  {
    name: "Front",
    prompt:
      "Show this exact object from the front view, straight-on perspective, keep the same object, same colors, same details, photorealistic, white background",
  },
  {
    name: "Front Right",
    prompt:
      "Show this exact object rotated 45 degrees to the right, front-right perspective view, keep the same object, same colors, same details, photorealistic, white background",
  },
  {
    name: "Right",
    prompt:
      "Show this exact object from the right side view, 90 degrees rotated, keep the same object, same colors, same details, photorealistic, white background",
  },
  {
    name: "Back Right",
    prompt:
      "Show this exact object from the back-right view, 135 degrees rotated, keep the same object, same colors, same details, photorealistic, white background",
  },
  {
    name: "Back",
    prompt:
      "Show this exact object from the back view, 180 degrees rotated, keep the same object, same colors, same details, photorealistic, white background",
  },
  {
    name: "Back Left",
    prompt:
      "Show this exact object from the back-left view, 225 degrees rotated, keep the same object, same colors, same details, photorealistic, white background",
  },
  {
    name: "Left",
    prompt:
      "Show this exact object from the left side view, 270 degrees rotated, keep the same object, same colors, same details, photorealistic, white background",
  },
  {
    name: "Front Left",
    prompt:
      "Show this exact object from the front-left view, 315 degrees rotated, keep the same object, same colors, same details, photorealistic, white background",
  },
  {
    name: "Top View",
    prompt:
      "Show this exact object from a top-down bird's eye view, looking straight down, keep the same object, same colors, same details, photorealistic, white background",
  },
];

/** Check if Puter.js SDK is loaded and available */
export function isPuterAvailable(): boolean {
  try {
    return (
      typeof puter !== "undefined" &&
      puter !== null &&
      typeof puter.ai !== "undefined" &&
      typeof puter.ai.txt2img === "function"
    );
  } catch {
    return false;
  }
}

/** Convert an HTMLImageElement to base64 data */
function imageElementToBase64(img: HTMLImageElement): Promise<string> {
  return new Promise((resolve, reject) => {
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth || img.width || 1024;
    canvas.height = img.naturalHeight || img.height || 1024;
    const ctx = canvas.getContext("2d");
    if (!ctx) {
      reject(new Error("Failed to get canvas context"));
      return;
    }
    ctx.drawImage(img, 0, 0);
    try {
      const dataUrl = canvas.toDataURL("image/png");
      const base64 = dataUrl.split(",")[1];
      resolve(base64);
    } catch (e) {
      reject(e);
    }
  });
}

/** Strip the data URL prefix to get raw base64 */
function stripDataUrlPrefix(dataUrl: string): string {
  const commaIndex = dataUrl.indexOf(",");
  if (commaIndex !== -1) {
    return dataUrl.substring(commaIndex + 1);
  }
  return dataUrl;
}

export interface PuterGenerationResult {
  name: string;
  success: boolean;
  image_data?: string;
  content_type?: string;
  error?: string;
}

/**
 * Generate a single angle image using Puter.js
 * @param inputImageBase64 - The input image as a data URL (data:image/...;base64,...)
 * @param angleConfig - The angle configuration with name and prompt
 * @param timeoutMs - Timeout in milliseconds (default: 60000)
 */
export async function generateAngleWithPuter(
  inputImageBase64: string,
  angleConfig: AngleConfig,
  timeoutMs: number = 60000
): Promise<PuterGenerationResult> {
  if (!isPuterAvailable()) {
    return {
      name: angleConfig.name,
      success: false,
      error: "Puter.js not available",
    };
  }

  try {
    const rawBase64 = stripDataUrlPrefix(inputImageBase64);

    const resultPromise = puter.ai.txt2img(angleConfig.prompt, {
      model: "gemini-2.5-flash-preview-image-generation",
      provider: "google-vertex",
      input_image: rawBase64,
    });

    const timeoutPromise = new Promise<never>((_, reject) => {
      setTimeout(
        () => reject(new Error("Puter.js generation timed out")),
        timeoutMs
      );
    });

    const imgElement = await Promise.race([resultPromise, timeoutPromise]);

    // Wait for the image to load if needed
    if (!imgElement.complete) {
      await new Promise<void>((resolve, reject) => {
        imgElement.onload = () => resolve();
        imgElement.onerror = () =>
          reject(new Error("Generated image failed to load"));
        setTimeout(() => reject(new Error("Image load timed out")), 10000);
      });
    }

    const base64Data = await imageElementToBase64(imgElement);

    return {
      name: angleConfig.name,
      success: true,
      image_data: base64Data,
      content_type: "image/png",
    };
  } catch (e) {
    const errorMsg = e instanceof Error ? e.message : "Unknown Puter.js error";
    return {
      name: angleConfig.name,
      success: false,
      error: errorMsg,
    };
  }
}

/**
 * Generate all angle images using Puter.js sequentially
 * This reduces load and is more reliable than parallel generation
 */
export async function generateAllAnglesWithPuter(
  inputImageBase64: string,
  onResult: (result: PuterGenerationResult, completed: number, total: number) => void,
  timeoutPerAngle: number = 60000
): Promise<PuterGenerationResult[]> {
  const results: PuterGenerationResult[] = [];
  const total = ANGLE_PROMPTS.length;

  for (let i = 0; i < ANGLE_PROMPTS.length; i++) {
    const angleConfig = ANGLE_PROMPTS[i];
    const result = await generateAngleWithPuter(
      inputImageBase64,
      angleConfig,
      timeoutPerAngle
    );
    results.push(result);
    onResult(result, i + 1, total);
  }

  return results;
}
