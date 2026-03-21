/**
 * Replicate provider — uses SDXL img2img for angle generation.
 * Requires VITE_REPLICATE_API_TOKEN environment variable.
 */

import type { ImageProvider, AngleParams, GenerationResult } from "./types";
import { ANGLE_PROMPTS, blobToBase64, withTimeout } from "./types";

const REPLICATE_API_URL = "https://api.replicate.com/v1/predictions";

export class ReplicateProvider implements ImageProvider {
  name = "Replicate";
  private apiToken: string;

  constructor() {
    this.apiToken = import.meta.env.VITE_REPLICATE_API_TOKEN || "";
  }

  isAvailable(): boolean {
    return this.apiToken.length > 0;
  }

  async generateAngle(
    imageBlob: Blob,
    params: AngleParams,
    timeoutMs: number,
  ): Promise<GenerationResult> {
    return withTimeout(this._generate(imageBlob, params), timeoutMs, this.name);
  }

  private async _generate(imageBlob: Blob, params: AngleParams): Promise<GenerationResult> {
    const base64Input = await blobToBase64(imageBlob);
    const dataUri = `data:image/png;base64,${base64Input}`;
    const prompt = ANGLE_PROMPTS[params.angleName] || `${params.angleName} view of this object`;

    // Create prediction using SDXL img2img
    const createResponse = await fetch(REPLICATE_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        version: "39ed52f2a78e934b3ba6e2a89f5b1c712de7dfea535525255b1aa35c5565e08b",
        input: {
          image: dataUri,
          prompt: `Generate a ${prompt}`,
          negative_prompt: "blurry, low quality, distorted",
          prompt_strength: 0.65,
          num_inference_steps: 25,
          guidance_scale: 7.5,
          width: 1024,
          height: 1024,
        },
      }),
    });

    if (!createResponse.ok) {
      const errText = await createResponse.text();
      throw new Error(`Replicate create failed: ${createResponse.status} - ${errText.substring(0, 200)}`);
    }

    const prediction = await createResponse.json();
    const statusUrl = prediction.urls?.get;
    if (!statusUrl) throw new Error("No status URL from Replicate");

    // Poll for completion
    const maxPolls = 60;
    for (let i = 0; i < maxPolls; i++) {
      await new Promise((r) => setTimeout(r, 2000));

      const pollResponse = await fetch(statusUrl, {
        headers: { Authorization: `Bearer ${this.apiToken}` },
      });
      if (!pollResponse.ok) continue;

      const pollResult = await pollResponse.json();
      if (pollResult.status === "succeeded" && pollResult.output) {
        const outputUrl = Array.isArray(pollResult.output) ? pollResult.output[0] : pollResult.output;
        const imgResponse = await fetch(outputUrl);
        if (!imgResponse.ok) throw new Error("Replicate image download failed");

        const imgBlob = await imgResponse.blob();
        const base64 = await blobToBase64(imgBlob);
        const contentType = imgResponse.headers.get("content-type") || "image/png";
        return { imageData: base64, contentType, provider: this.name };
      }

      if (pollResult.status === "failed" || pollResult.status === "canceled") {
        throw new Error(`Replicate prediction ${pollResult.status}: ${pollResult.error || "unknown"}`);
      }
    }

    throw new Error("Replicate prediction timed out during polling");
  }
}
