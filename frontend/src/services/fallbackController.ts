/**
 * Central fallback controller for multi-provider image generation.
 *
 * Tries providers in priority order:
 * 1. HuggingFace (primary — specialized camera angle model)
 * 2. Replicate (SDXL img2img, needs API key)
 * 3. Flux Schnell (public HF Space, fast generation)
 * 4. DeepAI (text2img, needs API key)
 * 5. Craiyon (public, lower quality, last resort)
 *
 * Each provider is retried 1-2 times before moving to the next.
 * Results are cached to avoid redundant API calls.
 */

import type { ImageProvider, AngleParams, GenerationResult } from "./providers/types";
import { HuggingFaceProvider } from "./providers/huggingface";
import { ReplicateProvider } from "./providers/replicate";
import { FluxSchnellProvider } from "./providers/fluxSchnell";
import { DeepAIProvider } from "./providers/deepai";
import { CraiyonProvider } from "./providers/craiyon";
import { imageCache } from "./imageCache";

const TIMEOUT_PER_PROVIDER_MS = 15000; // 15 seconds
const RETRIES_PER_PROVIDER = 2;
const RETRY_DELAY_MS = 1500;

export interface FallbackStatus {
  currentProvider: string;
  attempt: number;
  totalProviders: number;
  providerIndex: number;
}

export type OnFallbackStatus = (status: FallbackStatus) => void;

/** All providers in fallback order. */
function createProviders(): ImageProvider[] {
  return [
    new HuggingFaceProvider(),
    new ReplicateProvider(),
    new FluxSchnellProvider(),
    new DeepAIProvider(),
    new CraiyonProvider(),
  ];
}

const providers = createProviders();

/** Returns list of available provider names for UI display. */
export function getAvailableProviders(): string[] {
  return providers.filter((p) => p.isAvailable()).map((p) => p.name);
}

/**
 * Generate a single angle image using the fallback chain.
 * Tries each available provider with retries before moving to the next.
 */
export async function generateWithFallback(
  imageBlob: Blob,
  params: AngleParams,
  imageHash: string,
  lens: string,
  onStatus?: OnFallbackStatus,
): Promise<GenerationResult> {
  // Check cache first
  const cacheKey = imageCache.makeKey(imageHash, params.angleName, lens);
  const cached = imageCache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const availableProviders = providers.filter((p) => p.isAvailable());
  const errors: string[] = [];

  for (let pi = 0; pi < availableProviders.length; pi++) {
    const provider = availableProviders[pi];

    for (let attempt = 1; attempt <= RETRIES_PER_PROVIDER; attempt++) {
      onStatus?.({
        currentProvider: provider.name,
        attempt,
        totalProviders: availableProviders.length,
        providerIndex: pi,
      });

      try {
        const result = await provider.generateAngle(imageBlob, params, TIMEOUT_PER_PROVIDER_MS);
        // Cache successful result
        imageCache.set(cacheKey, result);
        return result;
      } catch (e) {
        const errMsg = e instanceof Error ? e.message : String(e);
        const label = `${provider.name} attempt ${attempt}/${RETRIES_PER_PROVIDER}`;
        console.warn(`[FallbackController] ${label} failed: ${errMsg}`);
        errors.push(`${label}: ${errMsg}`);

        // Wait before retry (but not after last attempt of last provider)
        const isLastAttempt = attempt === RETRIES_PER_PROVIDER;
        const isLastProvider = pi === availableProviders.length - 1;
        if (!(isLastAttempt && isLastProvider)) {
          await new Promise((r) => setTimeout(r, RETRY_DELAY_MS));
        }
      }
    }
  }

  throw new Error(
    `All providers failed for ${params.angleName}. Errors:\n${errors.join("\n")}`,
  );
}
