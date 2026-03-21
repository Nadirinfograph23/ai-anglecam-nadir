/**
 * HuggingFace Gradio Space provider — primary provider for camera angle generation.
 * Uses the Qwen Image Edit Angles space which is specifically designed for this task.
 */

import type { ImageProvider, AngleParams, GenerationResult } from "./types";
import { blobToBase64, withTimeout } from "./types";

const HF_SPACE_URL = "https://linoyts-qwen-image-edit-angles.hf.space";
const DEFAULT_GUIDANCE_SCALE = 1.0;
const DEFAULT_INFERENCE_STEPS = 4;
const DEFAULT_WIDTH = 1024;
const DEFAULT_HEIGHT = 1024;

function parseSSEResponse(text: string): string {
  const lines = text.split("\n");
  let errorMsg = "";

  for (const line of lines) {
    if (line.startsWith("event: error")) {
      errorMsg = "API returned an error";
    }
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
      } catch {
        continue;
      }
    }
  }

  throw new Error(errorMsg || "No result image found in SSE response");
}

export class HuggingFaceProvider implements ImageProvider {
  name = "HuggingFace";

  isAvailable(): boolean {
    return true; // Always available — public space, no API key needed
  }

  async generateAngle(
    imageBlob: Blob,
    params: AngleParams,
    timeoutMs: number,
  ): Promise<GenerationResult> {
    return withTimeout(this._generate(imageBlob, params), timeoutMs, this.name);
  }

  private async _generate(imageBlob: Blob, params: AngleParams): Promise<GenerationResult> {
    // Step 1: Upload image
    const uploadedPath = await this._uploadImage(imageBlob);

    // Step 2: Submit generation job
    const payload = {
      data: [
        false, // is_reset_val
        { path: uploadedPath, meta: { _type: "gradio.FileData" } },
        params.rotateDeg,
        params.moveForward,
        params.verticalTilt,
        params.wideangle,
        0, // seed
        true, // randomize_seed
        DEFAULT_GUIDANCE_SCALE,
        DEFAULT_INFERENCE_STEPS,
        DEFAULT_WIDTH,
        DEFAULT_HEIGHT,
        null, // prev_output
      ],
    };

    const submitResponse = await fetch(`${HF_SPACE_URL}/gradio_api/call/maybe_infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!submitResponse.ok) {
      const text = await submitResponse.text();
      throw new Error(`HF submit failed: ${submitResponse.status} - ${text.substring(0, 300)}`);
    }

    const submitResult = await submitResponse.json();
    const eventId = submitResult.event_id;
    if (!eventId) throw new Error("No event_id in HF submit response");

    // Step 3: Poll for result
    const resultResponse = await fetch(
      `${HF_SPACE_URL}/gradio_api/call/maybe_infer/${eventId}`,
    );
    if (!resultResponse.ok) {
      throw new Error(`HF result fetch failed: ${resultResponse.status}`);
    }

    const resultText = await resultResponse.text();
    const imageUrl = parseSSEResponse(resultText);

    // Step 4: Download generated image
    const imageResponse = await fetch(imageUrl);
    if (!imageResponse.ok) {
      throw new Error(`HF image download failed: ${imageResponse.status}`);
    }

    const contentType = imageResponse.headers.get("content-type") || "image/webp";
    const imageDataBlob = await imageResponse.blob();
    const base64 = await blobToBase64(imageDataBlob);

    return { imageData: base64, contentType, provider: this.name };
  }

  private async _uploadImage(imageBlob: Blob): Promise<string> {
    const formData = new FormData();
    formData.append("files", imageBlob, "input.png");

    const response = await fetch(`${HF_SPACE_URL}/gradio_api/upload`, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HF upload failed: ${response.status}`);
    }

    const result = await response.json();
    if (Array.isArray(result) && result.length > 0) {
      return result[0];
    }
    throw new Error(`Unexpected HF upload response: ${JSON.stringify(result)}`);
  }
}
