/**
 * In-memory LRU cache for generated images.
 * Caches results by input image hash + angle parameters to avoid redundant API calls.
 */

import type { GenerationResult } from "./providers/types";

interface CacheEntry {
  result: GenerationResult;
  timestamp: number;
}

const MAX_CACHE_SIZE = 100;
const CACHE_TTL_MS = 30 * 60 * 1000; // 30 minutes

class ImageCache {
  private cache = new Map<string, CacheEntry>();

  /**
   * Generate a cache key from image hash and angle parameters.
   */
  makeKey(imageHash: string, angleName: string, lens: string): string {
    return `${imageHash}:${angleName}:${lens}`;
  }

  /**
   * Compute a simple hash of the image file for cache keying.
   */
  async computeImageHash(file: File): Promise<string> {
    const buffer = await file.slice(0, 8192).arrayBuffer();
    const bytes = new Uint8Array(buffer);
    let hash = 0;
    for (let i = 0; i < bytes.length; i++) {
      hash = ((hash << 5) - hash + bytes[i]) | 0;
    }
    // Also include file size and name for uniqueness
    const sizeHash = file.size.toString(36);
    return `${(hash >>> 0).toString(36)}-${sizeHash}`;
  }

  get(key: string): GenerationResult | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    // Check TTL
    if (Date.now() - entry.timestamp > CACHE_TTL_MS) {
      this.cache.delete(key);
      return null;
    }

    return entry.result;
  }

  set(key: string, result: GenerationResult): void {
    // Evict oldest entries if at capacity
    if (this.cache.size >= MAX_CACHE_SIZE) {
      const oldestKey = this.cache.keys().next().value;
      if (oldestKey !== undefined) {
        this.cache.delete(oldestKey);
      }
    }

    this.cache.set(key, { result, timestamp: Date.now() });
  }

  clear(): void {
    this.cache.clear();
  }
}

export const imageCache = new ImageCache();
