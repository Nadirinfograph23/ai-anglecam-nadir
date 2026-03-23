import type { VercelRequest, VercelResponse } from "@vercel/node";

// ===== Configuration =====
const HF_SPACE_URLS = [
  "https://linoyts-qwen-image-edit-angles.hf.space",
  "https://linoyts-qwen-image-edit-2511-anypose.hf.space",
  "https://qwen-qwen-image-edit-2511.hf.space",
];

const REPLICATE_API_URL = "https://api.replicate.com/v1/predictions";
const STABLE_HORDE_API_URL = "https://stablehorde.net/api/v2";

const GENERATION_DEFAULTS = {
  guidanceScale: 1.0,
  inferenceSteps: 4,
  width: 1024,
  height: 1024,
};

const PROVIDER_TIMEOUT = 60_000; // 60s per provider (fits within Vercel function limits)
const UPLOAD_TIMEOUT = 30_000;

// ===== Simple in-memory cache (per cold-start) =====
const cache = new Map<string, { data: string; contentType: string; ts: number }>();
const CACHE_TTL = 600_000; // 10 min

function getCacheKey(imageHash: string, rotate: number, forward: number, tilt: number, wide: boolean): string {
  return `${imageHash}:${rotate}:${forward}:${tilt}:${wide}`;
}

// ===== Image hash =====
async function hashImage(buffer: ArrayBuffer): Promise<string> {
  const hashBuffer = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(hashBuffer))
    .map((b) => b.toString(16).padStart(2, "0"))
    .join("")
    .slice(0, 16);
}

// ===== Abort helper =====
function timeoutSignal(ms: number): AbortSignal {
  const controller = new AbortController();
  setTimeout(() => controller.abort(), ms);
  return controller.signal;
}

// ===== Provider 1: HuggingFace Inference API (Zero-1-to-3 style) =====
async function tryHuggingFaceInference(
  imageBase64: string,
  _rotateDeg: number,
  _moveForward: number,
  _verticalTilt: number,
  _wideangle: boolean,
): Promise<{ imageData: string; contentType: string } | null> {
  const hfToken = process.env.HF_API_TOKEN;
  if (!hfToken) return null;

  const models = [
    "sudo-ai/zero123plus-v1.1",
    "sudo-ai/zero123plus-v1.2",
  ];

  for (const model of models) {
    try {
      const response = await fetch(
        `https://api-inference.huggingface.co/models/${model}`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${hfToken}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ inputs: imageBase64 }),
          signal: timeoutSignal(PROVIDER_TIMEOUT),
        },
      );

      if (response.status === 503) {
        console.log(`[HF Inference] Model ${model} is loading, skipping...`);
        continue;
      }

      if (!response.ok) {
        console.log(`[HF Inference] Model ${model} returned ${response.status}`);
        continue;
      }

      const blob = await response.blob();
      const arrayBuffer = await blob.arrayBuffer();
      const base64 = Buffer.from(arrayBuffer).toString("base64");
      return { imageData: base64, contentType: blob.type || "image/png" };
    } catch (e) {
      console.warn(`[HF Inference] ${model} failed:`, (e as Error).message);
    }
  }
  return null;
}

// ===== Provider 2: HuggingFace Space (Gradio API - Qwen Image Edit Angles) =====
async function uploadToHFSpace(imageBuffer: Buffer, spaceUrl: string): Promise<string> {
  const formData = new FormData();
  const blob = new Blob([imageBuffer], { type: "image/png" });
  formData.append("files", blob, "input.png");

  const headers: Record<string, string> = {};
  if (process.env.HF_API_TOKEN) {
    headers["Authorization"] = `Bearer ${process.env.HF_API_TOKEN}`;
  }

  const response = await fetch(`${spaceUrl}/gradio_api/upload`, {
    method: "POST",
    body: formData,
    headers,
    signal: timeoutSignal(UPLOAD_TIMEOUT),
  });

  if (!response.ok) {
    const text = await response.text().catch(() => "");
    throw new Error(`Upload failed (${response.status}): ${text.slice(0, 200)}`);
  }

  const result = await response.json();
  if (Array.isArray(result) && result.length > 0) return result[0] as string;
  throw new Error("Unexpected upload response format");
}

function parseSSEForImageUrl(text: string): string {
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
        if (errorMsg && typeof data === "string") {
          throw new Error(data);
        }
        if (Array.isArray(data) && data.length > 0) {
          const first = data[0];
          if (first && typeof first === "object" && "url" in first) {
            return (first as { url: string }).url;
          }
        }
      } catch (e) {
        if (e instanceof Error && e.message !== "No image URL found") throw e;
        continue;
      }
    }
  }
  throw new Error(errorMsg || "No image URL found in API response");
}

// Detect the correct Gradio inference endpoint for a given space
async function detectInferEndpoint(spaceUrl: string): Promise<string> {
  try {
    const resp = await fetch(`${spaceUrl}/gradio_api/info`, {
      signal: timeoutSignal(10_000),
    });
    if (!resp.ok) return "/gradio_api/call/maybe_infer";
    const info = (await resp.json()) as { named_endpoints?: Record<string, unknown> };
    const endpoints = Object.keys(info.named_endpoints || {});
    if (endpoints.includes("/maybe_infer")) return "/gradio_api/call/maybe_infer";
    if (endpoints.includes("/infer")) return "/gradio_api/call/infer";
    if (endpoints.includes("/infer_edit_camera_angles")) return "/gradio_api/call/infer_edit_camera_angles";
    return "/gradio_api/call/maybe_infer";
  } catch {
    return "/gradio_api/call/maybe_infer";
  }
}

async function tryHFSpace(
  imageBuffer: Buffer,
  rotateDeg: number,
  moveForward: number,
  verticalTilt: number,
  wideangle: boolean,
): Promise<{ imageData: string; contentType: string } | null> {
  for (const spaceUrl of HF_SPACE_URLS) {
    try {
      console.log(`[HF Space] Trying ${spaceUrl}...`);

      // Quick health check — skip dead spaces immediately
      const healthResp = await fetch(`${spaceUrl}/gradio_api/info`, {
        signal: timeoutSignal(10_000),
      }).catch(() => null);
      if (!healthResp || !healthResp.ok) {
        console.warn(`[HF Space] ${spaceUrl} is not reachable (status: ${healthResp?.status ?? "network error"}), skipping`);
        continue;
      }

      const inferEndpoint = await detectInferEndpoint(spaceUrl);
      console.log(`[HF Space] Using endpoint ${inferEndpoint} for ${spaceUrl}`);

      const uploadedPath = await uploadToHFSpace(imageBuffer, spaceUrl);

      const headers: Record<string, string> = {
        "Content-Type": "application/json",
      };
      if (process.env.HF_API_TOKEN) {
        headers["Authorization"] = `Bearer ${process.env.HF_API_TOKEN}`;
      }

      const payload = {
        data: [
          false,
          { path: uploadedPath, meta: { _type: "gradio.FileData" } },
          rotateDeg,
          moveForward,
          verticalTilt,
          wideangle,
          0,
          true,
          GENERATION_DEFAULTS.guidanceScale,
          GENERATION_DEFAULTS.inferenceSteps,
          GENERATION_DEFAULTS.width,
          GENERATION_DEFAULTS.height,
          null,
        ],
      };

      const submitResponse = await fetch(
        `${spaceUrl}${inferEndpoint}`,
        {
          method: "POST",
          headers,
          body: JSON.stringify(payload),
          signal: timeoutSignal(PROVIDER_TIMEOUT),
        },
      );

      if (!submitResponse.ok) {
        const text = await submitResponse.text().catch(() => "");
        console.warn(`[HF Space] Submit failed on ${spaceUrl}: ${submitResponse.status} ${text.slice(0, 200)}`);
        continue;
      }

      const submitData = (await submitResponse.json()) as { event_id?: string };
      const eventId = submitData.event_id;
      if (!eventId) {
        console.warn(`[HF Space] No event_id from ${spaceUrl}`);
        continue;
      }

      const resultHeaders: Record<string, string> = {};
      if (process.env.HF_API_TOKEN) {
        resultHeaders["Authorization"] = `Bearer ${process.env.HF_API_TOKEN}`;
      }

      const resultResponse = await fetch(
        `${spaceUrl}${inferEndpoint}/${eventId}`,
        {
          headers: resultHeaders,
          signal: timeoutSignal(PROVIDER_TIMEOUT),
        },
      );

      if (!resultResponse.ok) {
        console.warn(`[HF Space] Result polling failed on ${spaceUrl}: ${resultResponse.status}`);
        continue;
      }

      const sseText = await resultResponse.text();
      const imageUrl = parseSSEForImageUrl(sseText);

      const imageResponse = await fetch(imageUrl, {
        headers: resultHeaders,
        signal: timeoutSignal(30_000),
      });

      if (!imageResponse.ok) {
        console.warn(`[HF Space] Image download failed: ${imageResponse.status}`);
        continue;
      }

      const imageBlob = await imageResponse.blob();
      const arrayBuffer = await imageBlob.arrayBuffer();
      const base64 = Buffer.from(arrayBuffer).toString("base64");
      const contentType = imageBlob.type || "image/webp";

      return { imageData: base64, contentType };
    } catch (e) {
      console.warn(`[HF Space] ${spaceUrl} failed:`, (e as Error).message);
    }
  }
  return null;
}

// ===== Provider 3: Replicate =====
async function tryReplicate(
  imageBase64: string,
  _rotateDeg: number,
  _moveForward: number,
  _verticalTilt: number,
  _wideangle: boolean,
): Promise<{ imageData: string; contentType: string } | null> {
  const replicateToken = process.env.REPLICATE_API_TOKEN;
  if (!replicateToken) return null;

  try {
    console.log("[Replicate] Starting prediction...");

    const createResponse = await fetch(REPLICATE_API_URL, {
      method: "POST",
      headers: {
        Authorization: `Token ${replicateToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        version: "75b33f253f7714a281ad3e9948b76aa131f602b3ecace31e5e93b32e5a115014",
        input: {
          image: `data:image/png;base64,${imageBase64}`,
        },
      }),
      signal: timeoutSignal(30_000),
    });

    if (!createResponse.ok) {
      console.warn(`[Replicate] Create failed: ${createResponse.status}`);
      return null;
    }

    const prediction = (await createResponse.json()) as {
      id: string;
      status: string;
      output?: string[];
      urls?: { get: string };
    };

    // Poll for result
    const pollUrl = prediction.urls?.get || `${REPLICATE_API_URL}/${prediction.id}`;
    const startTime = Date.now();

    while (Date.now() - startTime < PROVIDER_TIMEOUT) {
      await new Promise((r) => setTimeout(r, 3000));

      const pollResponse = await fetch(pollUrl, {
        headers: { Authorization: `Token ${replicateToken}` },
        signal: timeoutSignal(15_000),
      });

      if (!pollResponse.ok) continue;

      const result = (await pollResponse.json()) as {
        status: string;
        output?: string[];
        error?: string;
      };

      if (result.status === "succeeded" && result.output && result.output.length > 0) {
        const outputUrl = result.output[0];
        const imgResp = await fetch(outputUrl, { signal: timeoutSignal(30_000) });
        if (!imgResp.ok) return null;

        const imgBlob = await imgResp.blob();
        const arrBuf = await imgBlob.arrayBuffer();
        return {
          imageData: Buffer.from(arrBuf).toString("base64"),
          contentType: imgBlob.type || "image/png",
        };
      }

      if (result.status === "failed") {
        console.warn(`[Replicate] Prediction failed: ${result.error}`);
        return null;
      }
    }

    console.warn("[Replicate] Timed out waiting for result");
    return null;
  } catch (e) {
    console.warn("[Replicate] Error:", (e as Error).message);
    return null;
  }
}

// ===== Provider 4: Stable Horde =====
async function tryStableHorde(
  imageBase64: string,
  _rotateDeg: number,
  _moveForward: number,
  _verticalTilt: number,
  _wideangle: boolean,
): Promise<{ imageData: string; contentType: string } | null> {
  const hordeApiKey = process.env.STABLE_HORDE_API_KEY || "0000000000";

  try {
    console.log("[Stable Horde] Starting generation...");

    const createResponse = await fetch(`${STABLE_HORDE_API_URL}/generate/async`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        apikey: hordeApiKey,
      },
      body: JSON.stringify({
        prompt: "same object from a different camera angle, photorealistic",
        params: {
          sampler_name: "k_euler",
          cfg_scale: 7.5,
          width: 512,
          height: 512,
          steps: 30,
        },
        source_image: imageBase64,
        source_processing: "img2img",
        models: ["stable_diffusion"],
      }),
      signal: timeoutSignal(30_000),
    });

    if (!createResponse.ok) {
      console.warn(`[Stable Horde] Create failed: ${createResponse.status}`);
      return null;
    }

    const createResult = (await createResponse.json()) as { id?: string };
    if (!createResult.id) return null;

    const startTime = Date.now();
    while (Date.now() - startTime < PROVIDER_TIMEOUT) {
      await new Promise((r) => setTimeout(r, 5000));

      const checkResp = await fetch(
        `${STABLE_HORDE_API_URL}/generate/check/${createResult.id}`,
        {
          headers: { apikey: hordeApiKey },
          signal: timeoutSignal(15_000),
        },
      );

      if (!checkResp.ok) continue;

      const checkResult = (await checkResp.json()) as { done?: boolean; faulted?: boolean };

      if (checkResult.faulted) {
        console.warn("[Stable Horde] Generation faulted");
        return null;
      }

      if (checkResult.done) {
        const statusResp = await fetch(
          `${STABLE_HORDE_API_URL}/generate/status/${createResult.id}`,
          {
            headers: { apikey: hordeApiKey },
            signal: timeoutSignal(15_000),
          },
        );

        if (!statusResp.ok) return null;

        const statusResult = (await statusResp.json()) as {
          generations?: Array<{ img?: string }>;
        };

        if (statusResult.generations && statusResult.generations.length > 0) {
          const gen = statusResult.generations[0];
          if (gen.img) {
            return { imageData: gen.img, contentType: "image/png" };
          }
        }
        return null;
      }
    }

    console.warn("[Stable Horde] Timed out");
    return null;
  } catch (e) {
    console.warn("[Stable Horde] Error:", (e as Error).message);
    return null;
  }
}

// ===== Main handler =====
export default async function handler(req: VercelRequest, res: VercelResponse) {
  // CORS headers
  res.setHeader("Access-Control-Allow-Origin", "*");
  res.setHeader("Access-Control-Allow-Methods", "POST, OPTIONS");
  res.setHeader("Access-Control-Allow-Headers", "Content-Type");

  if (req.method === "OPTIONS") {
    return res.status(200).end();
  }

  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  try {
    const { imageData, rotateDeg, moveForward, verticalTilt, wideangle } = req.body as {
      imageData: string;
      rotateDeg: number;
      moveForward: number;
      verticalTilt: number;
      wideangle: boolean;
    };

    if (!imageData) {
      return res.status(400).json({ error: "imageData is required" });
    }

    const rotate = rotateDeg ?? 0;
    const forward = moveForward ?? 2.0;
    const tilt = verticalTilt ?? 0;
    const wide = wideangle ?? false;

    // Check cache
    const imageBuffer = Buffer.from(imageData, "base64");
    const imgHash = await hashImage(imageBuffer.buffer as ArrayBuffer);
    const cacheKey = getCacheKey(imgHash, rotate, forward, tilt, wide);
    const cached = cache.get(cacheKey);
    if (cached && Date.now() - cached.ts < CACHE_TTL) {
      console.log("[Cache] Hit for", cacheKey);
      return res.status(200).json({
        success: true,
        imageData: cached.data,
        contentType: cached.contentType,
        provider: "cache",
      });
    }

    // Run all providers in parallel with Promise.any for speed optimization.
    // The first provider to succeed wins; if all fail, return error.
    const providers: Array<{
      name: string;
      fn: () => Promise<{ imageData: string; contentType: string } | null>;
    }> = [
      {
        name: "HuggingFace Inference",
        fn: () => tryHuggingFaceInference(imageData, rotate, forward, tilt, wide),
      },
      {
        name: "HuggingFace Space",
        fn: () => tryHFSpace(imageBuffer, rotate, forward, tilt, wide),
      },
      {
        name: "Replicate",
        fn: () => tryReplicate(imageData, rotate, forward, tilt, wide),
      },
      {
        name: "Stable Horde",
        fn: () => tryStableHorde(imageData, rotate, forward, tilt, wide),
      },
    ];

    // Wrap each provider so null results become rejections for Promise.any
    const providerPromises = providers.map((provider) =>
      (async () => {
        console.log(`[Provider] Trying ${provider.name}...`);
        const result = await provider.fn();
        if (!result) {
          throw new Error(`${provider.name} returned no result`);
        }
        console.log(`[Provider] ${provider.name} succeeded`);
        return { ...result, providerName: provider.name };
      })(),
    );

    try {
      const winner = await Promise.any(providerPromises);

      // Cache successful result
      cache.set(cacheKey, { data: winner.imageData, contentType: winner.contentType, ts: Date.now() });
      // Limit cache size
      if (cache.size > 100) {
        const firstKey = cache.keys().next().value;
        if (firstKey) cache.delete(firstKey);
      }

      return res.status(200).json({
        success: true,
        imageData: winner.imageData,
        contentType: winner.contentType,
        provider: winner.providerName,
      });
    } catch (aggErr) {
      // All providers failed — log individual reasons
      if (aggErr instanceof AggregateError) {
        aggErr.errors.forEach((err, i) => {
          console.error(`[Provider] ${providers[i].name} failed:`, (err as Error).message);
        });
      } else {
        console.error("[Provider] All providers failed:", aggErr);
      }

      return res.status(502).json({
        success: false,
        error: "All providers failed. Please try again later.",
      });
    }
  } catch (e) {
    console.error("[Handler] Unexpected error:", e);
    return res.status(500).json({
      success: false,
      error: (e as Error).message || "Internal server error",
    });
  }
}
