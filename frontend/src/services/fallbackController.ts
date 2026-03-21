/**
 * Image generation controller using HuggingFace with robust retry logic.
 * Results are cached to avoid redundant API calls.
 */

import type { AngleParams, GenerationResult } from "./providers/types";
import { HuggingFaceProvider } from "./providers/huggingface";
import { imageCache } from "./imageCache";

const TIMEOUT_MS = 60000; // 60 seconds — HF spaces can be slow
const MAX_RETRIES = 3;
const RETRY_DELAY_MS = 2000;

const provider = new HuggingFaceProvider();

/**
 * Generate a single angle image with retries and caching.
 */
export async function generateWithFallback(
  imageBlob: Blob,
  params: AngleParams,
  imageHash: string,
  lens: string,
): Promise<GenerationResult> {
  // Check cache first
  const cacheKey = imageCache.makeKey(imageHash, params.angleName, lens);
  const cached = imageCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  let lastError: Error | null = null;

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const result = await provider.generateAngle(imageBlob, params, TIMEOUT_MS);
      imageCache.set(cacheKey, result);
      return result;
    } catch (e) {
      lastError = e instanceof Error ? e : new Error(String(e));
      console.warn(`[Generator] attempt ${attempt}/${MAX_RETRIES} failed for ${params.angleName}: ${lastError.message}`);

      if (attempt < MAX_RETRIES) {
        const delay = RETRY_DELAY_MS * Math.pow(1.5, attempt - 1);
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }

  throw new Error(
    `Failed to generate ${params.angleName} after ${MAX_RETRIES} attempts: ${lastError?.message}`,
  );
}
