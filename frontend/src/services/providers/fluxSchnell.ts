/**
 * Flux Schnell provider — uses a public HuggingFace Space for fast image generation.
 * No API key required.
 */

import type { ImageProvider, AngleParams, GenerationResult } from "./types";
import { ANGLE_PROMPTS, blobToBase64, withTimeout } from "./types";

const FLUX_SPACE_URL = "https://black-forest-labs-flux-1-schnell.hf.space";

function parseFluxSSE(text: string): string {
  const lines = text.split("\n");
  for (const line of lines) {
    if (line.startsWith("data: ")) {
      const dataStr = line.substring(6).trim();
      if (dataStr === "null") continue;
      try {
        const data = JSON.parse(dataStr);
        if (Array.isArray(data) && data.length > 0) {
          const first = data[0];
          if (typeof first === "object" && first !== null && "url" in first) {
            return first.url;
          }
        }
        // Some spaces return the image URL directly in the data
        if (typeof data === "string" && data.startsWith("http")) {
          return data;
        }
      } catch {
        continue;
      }
    }
  }
  throw new Error("No image URL found in Flux SSE response");
}

export class FluxSchnellProvider implements ImageProvider {
  name = "FluxSchnell";

  isAvailable(): boolean {
    return true; // Public HF Space, no key needed
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
    const fullPrompt = `A high quality ${prompt}, detailed, 8k resolution, studio lighting`;

    // Step 1: Submit to Flux Schnell
    const payload = {
      data: [
        fullPrompt,
        0,    // seed
        true, // randomize_seed
        1024, // width
        1024, // height
        4,    // num_inference_steps
      ],
    };

    const submitResponse = await fetch(`${FLUX_SPACE_URL}/gradio_api/call/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!submitResponse.ok) {
      const text = await submitResponse.text();
      throw new Error(`Flux submit failed: ${submitResponse.status} - ${text.substring(0, 200)}`);
    }

    const submitResult = await submitResponse.json();
    const eventId = submitResult.event_id;
    if (!eventId) throw new Error("No event_id from Flux Space");

    // Step 2: Get result
    const resultResponse = await fetch(
      `${FLUX_SPACE_URL}/gradio_api/call/infer/${eventId}`,
    );
    if (!resultResponse.ok) {
      throw new Error(`Flux result failed: ${resultResponse.status}`);
    }

    const resultText = await resultResponse.text();
    const imageUrl = parseFluxSSE(resultText);

    // Step 3: Download image
    const imageResponse = await fetch(imageUrl);
    if (!imageResponse.ok) {
      throw new Error(`Flux image download failed: ${imageResponse.status}`);
    }

    const contentType = imageResponse.headers.get("content-type") || "image/webp";
    const imgBlob = await imageResponse.blob();
    const base64 = await blobToBase64(imgBlob);

    return { imageData: base64, contentType, provider: this.name };
  }
}
