/**
 * Craiyon provider — last-resort fallback using the Craiyon (DALL-E Mini) API.
 * No API key required. Lower quality but always available.
 */

import type { ImageProvider, AngleParams, GenerationResult } from "./types";
import { ANGLE_PROMPTS, withTimeout } from "./types";

const CRAIYON_API_URL = "https://api.craiyon.com/v3";

export class CraiyonProvider implements ImageProvider {
  name = "Craiyon";

  isAvailable(): boolean {
    return true; // Public API, no key needed
  }

  async generateAngle(
    imageBlob: Blob,
    params: AngleParams,
    timeoutMs: number,
  ): Promise<GenerationResult> {
    return withTimeout(this._generate(imageBlob, params), timeoutMs, this.name);
  }

  private async _generate(_imageBlob: Blob, params: AngleParams): Promise<GenerationResult> {
    const prompt = ANGLE_PROMPTS[params.angleName] || `${params.angleName} view`;
    const fullPrompt = `A high quality ${prompt}, photorealistic, product photography`;

    const response = await fetch(CRAIYON_API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: fullPrompt,
        negative_prompt: "blurry, low quality, distorted, watermark",
        model: "photo",
        version: "c4ue22fb7kb6wlac",
        token: null,
      }),
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Craiyon failed: ${response.status} - ${errText.substring(0, 200)}`);
    }

    const result = await response.json();

    // Craiyon returns images as base64 in the response
    if (result.images && result.images.length > 0) {
      const imageBase64 = result.images[0];
      return {
        imageData: imageBase64,
        contentType: "image/webp",
        provider: this.name,
      };
    }

    throw new Error("No images returned from Craiyon");
  }
}
