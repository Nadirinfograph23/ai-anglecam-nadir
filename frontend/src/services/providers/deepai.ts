/**
 * DeepAI provider — uses the DeepAI image generation API.
 * Requires VITE_DEEPAI_API_KEY environment variable.
 */

import type { ImageProvider, AngleParams, GenerationResult } from "./types";
import { ANGLE_PROMPTS, blobToBase64, withTimeout } from "./types";

const DEEPAI_API_URL = "https://api.deepai.org/api/text2img";

export class DeepAIProvider implements ImageProvider {
  name = "DeepAI";
  private apiKey: string;

  constructor() {
    this.apiKey = import.meta.env.VITE_DEEPAI_API_KEY || "";
  }

  isAvailable(): boolean {
    return this.apiKey.length > 0;
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
    const fullPrompt = `A high quality ${prompt}, photorealistic, studio lighting, detailed`;

    const formData = new FormData();
    formData.append("text", fullPrompt);

    const response = await fetch(DEEPAI_API_URL, {
      method: "POST",
      headers: { "api-key": this.apiKey },
      body: formData,
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`DeepAI failed: ${response.status} - ${errText.substring(0, 200)}`);
    }

    const result = await response.json();
    const outputUrl = result.output_url;
    if (!outputUrl) {
      throw new Error("No output_url from DeepAI");
    }

    // Download generated image
    const imgResponse = await fetch(outputUrl);
    if (!imgResponse.ok) {
      throw new Error(`DeepAI image download failed: ${imgResponse.status}`);
    }

    const imgBlob = await imgResponse.blob();
    const base64 = await blobToBase64(imgBlob);
    const contentType = imgResponse.headers.get("content-type") || "image/png";

    return { imageData: base64, contentType, provider: this.name };
  }
}
