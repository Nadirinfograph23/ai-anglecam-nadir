import { useState, useRef, useCallback, useMemo, Component, type ReactNode, type ErrorInfo } from "react";
import {
  Camera,
  Upload,
  Download,
  RotateCcw,
  Loader2,
  AlertCircle,
  CheckCircle2,
  ImageIcon,
  Sparkles,
  X,
} from "lucide-react";

// Error Boundary for graceful error handling
class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean; error: Error | null }> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }
  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("App error:", error, errorInfo);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-gray-950 text-white flex items-center justify-center">
          <div className="text-center p-8">
            <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
            <h2 className="text-xl font-bold mb-2">Something went wrong</h2>
            <p className="text-gray-400 mb-4">{this.state.error?.message}</p>
            <button
              onClick={() => { this.setState({ hasError: false, error: null }); window.location.reload(); }}
              className="px-4 py-2 bg-blue-600 rounded-lg hover:bg-blue-500 transition-colors"
            >
              Reload App
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

// ===== Provider Types =====
type ProviderName = "puter" | "huggingface";

interface ProviderStatus {
  name: ProviderName;
  label: string;
  available: boolean;
  failCount: number;
}

// ===== Constants =====
const HF_SPACE_URL = "https://linoyts-qwen-image-edit-angles.hf.space";

const PUTER_MODELS_FALLBACK = [
  { model: "dall-e-3", provider: "openai-image-generation" },
  { model: "gpt-image-1-mini", provider: "openai-image-generation" },
  { model: "flux-1-schnell", provider: "together" },
  { model: "grok-2-image", provider: "xai" },
];

const ANGLE_PROMPTS: Record<string, string> = {
  "Front": "viewed from the front, facing the camera directly, centered frontal view",
  "Front Right": "viewed from the front-right at a 45-degree angle, three-quarter view from the right",
  "Right": "viewed from the right side, 90-degree side profile view from the right",
  "Back Right": "viewed from the back-right at a 135-degree angle, three-quarter rear view from the right",
  "Back": "viewed from behind, rear view showing the back, 180-degree turn",
  "Back Left": "viewed from the back-left at a 135-degree angle, three-quarter rear view from the left",
  "Left": "viewed from the left side, 90-degree side profile view from the left",
  "Front Left": "viewed from the front-left at a 45-degree angle, three-quarter view from the left",
  "Top View": "viewed from above, bird's-eye view looking down at a 60-degree angle, top-down perspective",
};

const PREDEFINED_ANGLES = [
  { name: "Front", h: 0, v: 0 },
  { name: "Front Right", h: 45, v: 0 },
  { name: "Right", h: 90, v: 0 },
  { name: "Back Right", h: 135, v: 0 },
  { name: "Back", h: 180, v: 0 },
  { name: "Back Left", h: -135, v: 0 },
  { name: "Left", h: -90, v: 0 },
  { name: "Front Left", h: -45, v: 0 },
  { name: "Top View", h: 0, v: 60 },
];

const GENERATION_DEFAULTS = {
  guidanceScale: 1.0,
  inferenceSteps: 4,
  width: 1024,
  height: 1024,
};

// ===== Angle Conversion Helpers =====
function clampRotate(deg: number): number {
  deg = deg % 360;
  if (deg > 180) deg -= 360;
  if (deg < -180) deg += 360;
  return deg;
}

function convertVertical(v: number): number {
  return Math.max(-1.0, Math.min(1.0, v / 60.0));
}

function convertForward(lens: string): number {
  const mapping: Record<string, number> = { closeup: 5.0, wide: 0.0, normal: 2.0 };
  return mapping[lens] ?? 2.0;
}

// ===== Image Optimization =====
async function optimizeImage(file: File, maxSize = 2048): Promise<Blob> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const objectUrl = URL.createObjectURL(file);
    img.onload = () => {
      URL.revokeObjectURL(objectUrl);
      const canvas = document.createElement("canvas");
      let { width, height } = img;
      if (Math.max(width, height) > maxSize) {
        const ratio = maxSize / Math.max(width, height);
        width = Math.round(width * ratio);
        height = Math.round(height * ratio);
      }
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) { reject(new Error("Canvas context unavailable")); return; }
      ctx.drawImage(img, 0, 0, width, height);
      canvas.toBlob(
        (blob) => blob ? resolve(blob) : reject(new Error("Image conversion failed")),
        "image/png"
      );
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      reject(new Error("Failed to load image"));
    };
    img.src = objectUrl;
  });
}

function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

function imgElementToBase64(imgEl: HTMLImageElement): { imageData: string; contentType: string } {
  const canvas = document.createElement("canvas");
  canvas.width = imgEl.naturalWidth || imgEl.width || 1024;
  canvas.height = imgEl.naturalHeight || imgEl.height || 1024;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Canvas context unavailable");
  ctx.drawImage(imgEl, 0, 0);
  const dataUrl = canvas.toDataURL("image/png");
  const base64 = dataUrl.split(",")[1] || "";
  return { imageData: base64, contentType: "image/png" };
}

// ===== Puter.js Provider =====
function isPuterAvailable(): boolean {
  return typeof puter !== "undefined" && puter?.ai !== undefined;
}

async function analyzeImageWithPuter(imageDataUrl: string): Promise<string> {
  if (!isPuterAvailable()) throw new Error("Puter.js not available");

  const description = await puter.ai.chat(
    "Describe this image in detail for image generation. Focus on: the main subject, its shape, colors, materials, textures, lighting, background, and overall composition. Be specific and concise. Output ONLY the description, nothing else.",
    imageDataUrl,
    { model: "qwen/qwen2.5-vl-72b-instruct" }
  );

  if (!description || description.length < 10) {
    throw new Error("Failed to get image description from Puter.js");
  }

  return description;
}

async function generateWithPuter(
  description: string,
  angleName: string,
): Promise<{ imageData: string; contentType: string }> {
  if (!isPuterAvailable()) throw new Error("Puter.js not available");

  const anglePrompt = ANGLE_PROMPTS[angleName] || "viewed from a different angle";
  const fullPrompt = description + ", " + anglePrompt + ", photorealistic, high quality, detailed, same subject and style as original";

  let lastError: Error | null = null;

  for (const modelConfig of PUTER_MODELS_FALLBACK) {
    try {
      const imgElement = await puter.ai.txt2img({
        prompt: fullPrompt,
        model: modelConfig.model,
        provider: modelConfig.provider,
      });

      if (imgElement instanceof HTMLImageElement) {
        await new Promise<void>((resolve, reject) => {
          if (imgElement.complete && imgElement.naturalWidth > 0) {
            resolve();
            return;
          }
          imgElement.onload = () => resolve();
          imgElement.onerror = () => reject(new Error("Image failed to load"));
          setTimeout(() => reject(new Error("Image load timeout")), 30000);
        });

        return imgElementToBase64(imgElement);
      }

      throw new Error("Unexpected response from Puter.js txt2img");
    } catch (e) {
      lastError = e instanceof Error ? e : new Error(String(e));
      console.warn("[Puter] Model " + modelConfig.model + " failed:", lastError.message);
      continue;
    }
  }

  throw lastError || new Error("All Puter.js models failed");
}

// ===== HuggingFace Gradio API Client =====
async function uploadToHF(imageBlob: Blob): Promise<string> {
  const formData = new FormData();
  formData.append("files", imageBlob, "input.png");

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);

  try {
    const response = await fetch(HF_SPACE_URL + "/gradio_api/upload", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error("Upload failed (" + response.status + "): " + text.slice(0, 200));
    }

    const result = await response.json();
    if (Array.isArray(result) && result.length > 0) return result[0];
    throw new Error("Unexpected upload response format");
  } finally {
    clearTimeout(timeout);
  }
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

function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result.split(",")[1] || "");
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

async function generateSingleAngleFromHF(
  uploadedPath: string,
  rotateDeg: number,
  moveForward: number,
  verticalTilt: number,
  wideangle: boolean,
): Promise<{ imageData: string; contentType: string }> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120000);

  try {
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

    const submitResponse = await fetch(HF_SPACE_URL + "/gradio_api/call/maybe_infer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });

    if (!submitResponse.ok) {
      const text = await submitResponse.text().catch(() => "");
      throw new Error("Generation submit failed (" + submitResponse.status + "): " + text.slice(0, 200));
    }

    const submitData = await submitResponse.json();
    const eventId = submitData.event_id;
    if (!eventId) throw new Error("No event_id in API response");

    const resultResponse = await fetch(
      HF_SPACE_URL + "/gradio_api/call/maybe_infer/" + eventId,
      { signal: controller.signal }
    );

    if (!resultResponse.ok) {
      throw new Error("Result polling failed (" + resultResponse.status + ")");
    }

    const sseText = await resultResponse.text();
    const imageUrl = parseSSEForImageUrl(sseText);

    const imageResponse = await fetch(imageUrl, { signal: controller.signal });
    if (!imageResponse.ok) {
      throw new Error("Image download failed (" + imageResponse.status + ")");
    }

    const blob = await imageResponse.blob();
    const contentType = blob.type || "image/webp";
    const imageData = await blobToBase64(blob);

    return { imageData, contentType };
  } finally {
    clearTimeout(timeout);
  }
}

async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries = 3,
  baseDelay = 3000,
  label = "",
): Promise<T> {
  let lastError: Error | null = null;
  for (let attempt = 0; attempt < maxRetries; attempt++) {
    try {
      return await fn();
    } catch (e) {
      lastError = e instanceof Error ? e : new Error(String(e));
      console.warn("[Retry] " + label + " attempt " + (attempt + 1) + "/" + maxRetries + " failed:", lastError.message);
      if (attempt < maxRetries - 1) {
        const delay = Math.min(baseDelay * Math.pow(2, attempt), 30000);
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }
  throw lastError || new Error(label + " failed after " + maxRetries + " attempts");
}

// ===== Multi-Provider Generation Engine =====
async function generateAngleWithFallback(
  angleName: string,
  imageDescription: string | null,
  lens: string,
  hfUploadedPath: string | null,
  providers: ProviderStatus[],
  onProviderFail: (name: ProviderName) => void,
): Promise<{ imageData: string; contentType: string; usedProvider: ProviderName }> {

  const sortedProviders = [...providers]
    .filter((p) => p.available)
    .sort((a, b) => a.failCount - b.failCount);

  for (const provider of sortedProviders) {
    try {
      if (provider.name === "puter") {
        if (!isPuterAvailable() || !imageDescription) {
          onProviderFail("puter");
          continue;
        }
        const result = await withRetry(
          () => generateWithPuter(imageDescription, angleName),
          2, 2000, "Puter/" + angleName
        );
        return { ...result, usedProvider: "puter" };
      }

      if (provider.name === "huggingface") {
        if (!hfUploadedPath) {
          onProviderFail("huggingface");
          continue;
        }
        const angle = PREDEFINED_ANGLES.find((a) => a.name === angleName);
        if (!angle) throw new Error("Unknown angle: " + angleName);

        const rotate = clampRotate(angle.h);
        const forward = convertForward(lens);
        const tilt = convertVertical(angle.v);
        const isWide = lens === "wide";

        const result = await withRetry(
          () => generateSingleAngleFromHF(hfUploadedPath, rotate, forward, tilt, isWide),
          3, 3000, "HF/" + angleName
        );
        return { ...result, usedProvider: "huggingface" };
      }
    } catch (e) {
      console.warn("[Fallback] Provider " + provider.name + " failed for " + angleName + ":", e instanceof Error ? e.message : String(e));
      onProviderFail(provider.name);
      continue;
    }
  }

  throw new Error("All providers failed for angle: " + angleName);
}

// ===== React Types and Constants =====
interface AngleResult {
  name: string;
  success: boolean;
  image_data?: string;
  content_type?: string;
  error?: string;
  provider?: ProviderName;
}

interface PendingAngle {
  name: string;
  success: false;
  pending: true;
}

type GridItem = AngleResult | PendingAngle;

const ANGLE_NAMES = [
  "Front", "Front Right", "Right", "Back Right", "Back",
  "Back Left", "Left", "Front Left", "Top View",
];

const LENS_OPTIONS = [
  { value: "normal", label: "Normal" },
  { value: "wide", label: "Wide Angle" },
  { value: "closeup", label: "Close-Up" },
];

function isPending(item: GridItem): item is PendingAngle {
  return "pending" in item && item.pending === true;
}

function getProviderBadgeColor(provider?: ProviderName): string {
  if (provider === "puter") return "bg-green-600/80";
  if (provider === "huggingface") return "bg-yellow-600/80";
  return "bg-gray-600/80";
}

function getProviderLabel(provider?: ProviderName): string {
  if (provider === "puter") return "Puter.js";
  if (provider === "huggingface") return "HuggingFace";
  return "";
}

function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [lens, setLens] = useState("normal");
  const [isGenerating, setIsGenerating] = useState(false);
  const [results, setResults] = useState<AngleResult[]>([]);
  const [progress, setProgress] = useState({ completed: 0, total: 9 });
  const [error, setError] = useState<string | null>(null);
  const [retryingAngle, setRetryingAngle] = useState<string | null>(null);
  const [retryingAll, setRetryingAll] = useState(false);
  const [imageCount, setImageCount] = useState<number | null>(null);
  const [activeProvider, setActiveProvider] = useState<string>("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const hfUploadedPathRef = useRef<string | null>(null);
  const imageDescriptionRef = useRef<string | null>(null);
  const imageDataUrlRef = useRef<string | null>(null);

  const [providerStatuses, setProviderStatuses] = useState<ProviderStatus[]>([
    { name: "puter", label: "Puter.js", available: true, failCount: 0 },
    { name: "huggingface", label: "HuggingFace", available: true, failCount: 0 },
  ]);

  const handleProviderFail = useCallback((name: ProviderName) => {
    setProviderStatuses((prev) =>
      prev.map((p) =>
        p.name === name ? { ...p, failCount: p.failCount + 1 } : p
      )
    );
  }, []);

  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file (JPEG, PNG, WebP)");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError("Image too large. Maximum size is 20MB.");
      return;
    }
    setImageFile(file);
    setError(null);
    setResults([]);
    hfUploadedPathRef.current = null;
    imageDescriptionRef.current = null;
    imageDataUrlRef.current = null;
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      setSelectedImage(dataUrl);
      imageDataUrlRef.current = dataUrl;
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const selectedAngleCount = imageCount || 9;

  const generateAllAngles = async () => {
    if (!imageFile || !imageCount) return;
    setIsGenerating(true);
    setError(null);
    setResults([]);
    const total = imageCount;
    setProgress({ completed: 0, total });

    // Reset provider fail counts
    setProviderStatuses((prev) =>
      prev.map((p) => ({ ...p, failCount: 0 }))
    );

    try {
      // Phase 1: Prepare providers in parallel
      setActiveProvider("Preparing providers...");

      const imageDataUrl = imageDataUrlRef.current || await fileToDataUrl(imageFile);
      imageDataUrlRef.current = imageDataUrl;

      const [puterReady, hfReady] = await Promise.allSettled([
        // Puter.js: Analyze image to get description
        (async () => {
          if (!isPuterAvailable()) throw new Error("Puter.js not loaded");
          setActiveProvider("Analyzing image with AI...");
          const description = await analyzeImageWithPuter(imageDataUrl);
          imageDescriptionRef.current = description;
          console.log("[Puter] Image description:", description.substring(0, 100) + "...");
          return true;
        })(),
        // HuggingFace: Upload image
        (async () => {
          const optimized = await optimizeImage(imageFile);
          const uploadedPath = await withRetry(
            () => uploadToHF(optimized),
            3, 3000, "HF Upload"
          );
          hfUploadedPathRef.current = uploadedPath;
          return true;
        })(),
      ]);

      const puterAvailable = puterReady.status === "fulfilled";
      const hfAvailable = hfReady.status === "fulfilled";

      if (!puterAvailable && !hfAvailable) {
        throw new Error("All providers failed to initialize. Please try again.");
      }

      // Update provider availability
      setProviderStatuses((prev) =>
        prev.map((p) => {
          if (p.name === "puter") return { ...p, available: puterAvailable };
          if (p.name === "huggingface") return { ...p, available: hfAvailable };
          return p;
        })
      );

      if (puterAvailable) {
        setActiveProvider("Puter.js (primary)");
      } else {
        setActiveProvider("HuggingFace (fallback)");
      }

      // Phase 2: Generate images
      let completedCount = 0;
      const anglesToGenerate = PREDEFINED_ANGLES.slice(0, total);
      const batchSize = 2;

      for (let i = 0; i < anglesToGenerate.length; i += batchSize) {
        const batch = anglesToGenerate.slice(i, i + batchSize);

        const batchPromises = batch.map(async (angle) => {
          try {
            const result = await generateAngleWithFallback(
              angle.name,
              imageDescriptionRef.current,
              lens,
              hfUploadedPathRef.current,
              providerStatuses,
              handleProviderFail,
            );

            completedCount++;
            setProgress({ completed: completedCount, total });
            setResults((prev) => [
              ...prev.filter((r) => r.name !== angle.name),
              {
                name: angle.name,
                success: true,
                image_data: result.imageData,
                content_type: result.contentType,
                provider: result.usedProvider,
              },
            ]);
          } catch (e) {
            completedCount++;
            setProgress({ completed: completedCount, total });
            setResults((prev) => [
              ...prev.filter((r) => r.name !== angle.name),
              {
                name: angle.name,
                success: false,
                error: e instanceof Error ? e.message : "Generation failed",
              },
            ]);
          }
        });

        await Promise.all(batchPromises);

        if (i + batchSize < anglesToGenerate.length) {
          await new Promise((r) => setTimeout(r, 1000));
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed. Please try again.");
    } finally {
      setIsGenerating(false);
      setActiveProvider("");
    }
  };

  const retryAngle = async (angleName: string) => {
    if (!imageFile) return;
    setRetryingAngle(angleName);
    setError(null);

    try {
      const imageDataUrl = imageDataUrlRef.current || await fileToDataUrl(imageFile);

      if (isPuterAvailable() && !imageDescriptionRef.current) {
        try {
          imageDescriptionRef.current = await analyzeImageWithPuter(imageDataUrl);
        } catch {
          // Continue without puter
        }
      }

      if (!hfUploadedPathRef.current) {
        try {
          const optimized = await optimizeImage(imageFile);
          hfUploadedPathRef.current = await withRetry(() => uploadToHF(optimized), 3, 3000, "Upload");
        } catch {
          // Continue without HF
        }
      }

      const result = await generateAngleWithFallback(
        angleName,
        imageDescriptionRef.current,
        lens,
        hfUploadedPathRef.current,
        providerStatuses,
        handleProviderFail,
      );

      setResults((prev) => [
        ...prev.filter((r) => r.name !== angleName),
        {
          name: angleName,
          success: true,
          image_data: result.imageData,
          content_type: result.contentType,
          provider: result.usedProvider,
        },
      ]);
    } catch (e) {
      setError("Retry for " + angleName + " failed: " + (e instanceof Error ? e.message : "Unknown error"));
    } finally {
      setRetryingAngle(null);
    }
  };

  const retryAllFailed = async () => {
    if (!imageFile) return;
    const failedResults = results.filter((r) => !r.success);
    if (failedResults.length === 0) return;
    setRetryingAll(true);
    setError(null);

    try {
      const imageDataUrl = imageDataUrlRef.current || await fileToDataUrl(imageFile);

      if (isPuterAvailable() && !imageDescriptionRef.current) {
        try {
          imageDescriptionRef.current = await analyzeImageWithPuter(imageDataUrl);
        } catch {
          // Continue
        }
      }

      if (!hfUploadedPathRef.current) {
        try {
          const optimized = await optimizeImage(imageFile);
          hfUploadedPathRef.current = await withRetry(() => uploadToHF(optimized), 3, 3000, "Upload");
        } catch {
          // Continue
        }
      }

      for (const failed of failedResults) {
        try {
          const result = await generateAngleWithFallback(
            failed.name,
            imageDescriptionRef.current,
            lens,
            hfUploadedPathRef.current,
            providerStatuses,
            handleProviderFail,
          );

          setResults((prev) => [
            ...prev.filter((r) => r.name !== failed.name),
            {
              name: failed.name,
              success: true,
              image_data: result.imageData,
              content_type: result.contentType,
              provider: result.usedProvider,
            },
          ]);
        } catch {
          // Continue with next angle
        }
      }
    } catch (e) {
      setError("Retry failed: " + (e instanceof Error ? e.message : "Unknown error"));
    } finally {
      setRetryingAll(false);
    }
  };

  const downloadImage = (result: AngleResult) => {
    if (!result.image_data || !result.content_type) return;
    const ext = result.content_type.includes("webp") ? "webp" : "png";
    const link = document.createElement("a");
    link.href = "data:" + result.content_type + ";base64," + result.image_data;
    link.download = "angle-" + result.name.toLowerCase().replace(/\s+/g, "-") + "." + ext;
    link.click();
  };

  const downloadAll = () => {
    const successResults = results.filter((r) => r.success);
    successResults.forEach((r, i) => {
      setTimeout(() => downloadImage(r), i * 300);
    });
  };

  const successCount = useMemo(() => results.filter((r) => r.success).length, [results]);
  const failCount = useMemo(() => results.filter((r) => !r.success).length, [results]);

  const activeAngleNames = useMemo(() => ANGLE_NAMES.slice(0, selectedAngleCount), [selectedAngleCount]);

  const getGridItems = useCallback((): GridItem[] => {
    if (isGenerating) {
      return activeAngleNames.map((name) => {
        const existing = results.find((r) => r.name === name);
        if (existing) return existing;
        return { name, success: false as const, pending: true as const };
      });
    }
    // Show all results (successful and failed) so user can retry failed ones
    return activeAngleNames
      .map((name) => results.find((r) => r.name === name))
      .filter((r): r is AngleResult => r !== undefined);
  }, [isGenerating, results, activeAngleNames]);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <header className="sticky top-0 z-30 w-full border-b border-gray-800/50 bg-gray-950/90 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 p-1.5">
                <Camera className="h-5 w-5 text-white" />
              </div>
              <div>
                <h1 className="text-base font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                  AI AngleCam Nadir
                </h1>
                <p className="text-xs text-gray-500 -mt-0.5 hidden sm:block">
                  Multi-Angle Image Generator
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <Sparkles className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Powered by Puter.js + HuggingFace</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white">Generate Multi-Angle Views</h2>
          <p className="text-gray-400 text-sm mt-1">
            Upload an image and generate different viewing angles automatically
          </p>
        </div>

        {/* Provider Status Bar */}
        <div className="mb-4 flex flex-wrap gap-2">
          {providerStatuses.map((p) => (
            <div
              key={p.name}
              className={"flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium " + (
                p.available && p.failCount === 0
                  ? "bg-green-900/40 text-green-400 border border-green-800/50"
                  : p.available && p.failCount > 0
                  ? "bg-yellow-900/40 text-yellow-400 border border-yellow-800/50"
                  : "bg-red-900/40 text-red-400 border border-red-800/50"
              )}
            >
              <span className={"inline-block w-1.5 h-1.5 rounded-full " + (
                p.available && p.failCount === 0 ? "bg-green-400" : p.available ? "bg-yellow-400" : "bg-red-400"
              )} />
              {p.label}
              {p.failCount > 0 && <span className="opacity-70">({p.failCount} fails)</span>}
            </div>
          ))}
          {activeProvider && (
            <div className="flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-medium bg-blue-900/40 text-blue-400 border border-blue-800/50">
              <Loader2 className="h-3 w-3 animate-spin" />
              {activeProvider}
            </div>
          )}
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="space-y-5">
            <div className="rounded-2xl bg-gray-900/60 border border-gray-800/50 p-5">
              <div className="flex items-center gap-2 mb-3">
                <h3 className="text-white font-semibold">Input Image</h3>
                <span className="text-red-500">*</span>
              </div>
              <div onDrop={handleDrop} onDragOver={(e) => e.preventDefault()} className="space-y-3">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className={"w-full flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-6 transition-all " + (
                    selectedImage
                      ? "border-gray-700 bg-gray-800/30 hover:border-cyan-500/50"
                      : "border-gray-600 bg-gray-800/50 hover:border-cyan-400/50 hover:bg-gray-800"
                  )}
                >
                  <Upload className="h-8 w-8 text-gray-400" />
                  <span className="text-sm text-gray-300 font-medium">
                    {selectedImage ? "Change Image" : "Upload Image"}
                  </span>
                  <span className="text-xs text-gray-500">JPEG, PNG, WebP (max 20MB)</span>
                </button>
                {selectedImage && (
                  <div className="relative rounded-xl border border-gray-700 bg-gray-800/30 overflow-hidden">
                    <img src={selectedImage} alt="Input preview" className="w-full h-auto max-h-64 object-contain" />
                    <button
                      onClick={() => { setSelectedImage(null); setImageFile(null); setResults([]); }}
                      className="absolute top-2 right-2 p-1 rounded-full bg-gray-900/80 hover:bg-red-600 transition-colors"
                    >
                      <X className="h-4 w-4" />
                    </button>
                  </div>
                )}
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={(e) => { const file = e.target.files?.[0]; if (file) handleFileSelect(file); }}
                  className="hidden"
                />
              </div>
            </div>

            <div className="rounded-2xl bg-gray-900/60 border border-gray-800/50 p-5">
              <h3 className="text-white font-semibold mb-3">Lens Type</h3>
              <div className="flex gap-2">
                {LENS_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setLens(opt.value)}
                    className={"px-4 py-2 rounded-lg text-sm font-medium transition-all " + (
                      lens === opt.value
                        ? "bg-white text-gray-900"
                        : "bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-700"
                    )}
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            </div>

            <div className="rounded-2xl bg-gray-900/60 border border-gray-800/50 p-5">
              <div className="flex items-center gap-2 mb-3">
                <h3 className="text-white font-semibold">Number of Images</h3>
                <span className="text-red-500">*</span>
              </div>
              <div className="flex flex-wrap gap-2">
                {[3, 4, 5, 6, 7, 8, 9].map((count) => (
                  <button
                    key={count}
                    onClick={() => setImageCount(count)}
                    className={"px-4 py-2 rounded-lg text-sm font-medium transition-all min-w-[3rem] " + (
                      imageCount === count
                        ? "bg-white text-gray-900"
                        : "bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-700"
                    )}
                  >
                    {count}
                  </button>
                ))}
              </div>
              {!imageCount && (
                <p className="text-amber-400 text-xs mt-2 flex items-center gap-1">
                  <AlertCircle className="h-3 w-3" />
                  Please select the number of images to generate
                </p>
              )}
            </div>

            <button
              onClick={generateAllAngles}
              disabled={!imageFile || !imageCount || isGenerating}
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-3.5 text-base font-semibold text-white shadow-lg shadow-blue-500/20 transition-all hover:shadow-blue-500/30 hover:from-blue-500 hover:to-blue-600 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  {"Generating... (" + progress.completed + "/" + progress.total + ")"}
                </>
              ) : (
                <>
                  <Sparkles className="h-5 w-5" />
                  {imageCount ? "Generate " + imageCount + " Angles" : "Generate Angles"}
                </>
              )}
            </button>

            {isGenerating && (
              <div className="rounded-xl bg-gray-800 p-3">
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-cyan-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: ((progress.completed / progress.total) * 100) + "%" }}
                  />
                </div>
                <p className="text-center text-gray-400 text-xs mt-2">
                  {progress.completed + "/" + progress.total + " angles completed"}
                </p>
              </div>
            )}

            {error && (
              <div className="rounded-xl bg-red-900/30 border border-red-800/50 p-3 flex items-start gap-2">
                <AlertCircle className="h-4 w-4 text-red-400 mt-0.5 shrink-0" />
                <p className="text-red-300 text-sm">{error}</p>
              </div>
            )}

            {results.length > 0 && !isGenerating && (
              <div className="space-y-2">
                <div className="rounded-xl bg-gray-800/50 p-3 flex items-center justify-between text-sm">
                  <div className="flex items-center gap-4">
                    <span className="flex items-center gap-1 text-green-400">
                      <CheckCircle2 className="h-4 w-4" />
                      {successCount + " succeeded"}
                    </span>
                    {failCount > 0 && (
                      <span className="flex items-center gap-1 text-red-400">
                        <AlertCircle className="h-4 w-4" />
                        {failCount + " failed"}
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-2">
                    {failCount > 0 && (
                      <button
                        onClick={retryAllFailed}
                        disabled={retryingAll}
                        className="flex items-center gap-1 px-3 py-1 rounded-lg text-xs font-medium bg-orange-600/80 text-white hover:bg-orange-500 transition-colors disabled:opacity-50"
                      >
                        {retryingAll ? (
                          <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        ) : (
                          <RotateCcw className="h-3.5 w-3.5" />
                        )}
                        Retry Failed
                      </button>
                    )}
                    {successCount > 0 && (
                      <button
                        onClick={downloadAll}
                        className="flex items-center gap-1 px-3 py-1 rounded-lg text-xs font-medium bg-gradient-to-r from-blue-600 to-cyan-600 text-white hover:from-blue-500 hover:to-cyan-500 transition-colors"
                      >
                        <Download className="h-3.5 w-3.5" />
                        Download All
                      </button>
                    )}
                  </div>
                </div>
                {failCount > 0 && (
                  <div className="rounded-xl bg-amber-900/30 border border-amber-700/40 p-3 flex items-start gap-2">
                    <AlertCircle className="h-4 w-4 text-amber-400 mt-0.5 shrink-0" />
                    <p className="text-amber-300 text-sm">
                      {"Some angles failed. Click 'Retry Failed' to try again with automatic provider fallback."}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>

          <div className="lg:col-span-2">
            <div className="rounded-2xl bg-gray-900/60 border border-gray-800/50 p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-white font-semibold">
                  {"Generated Angles" + (results.length > 0 ? " (" + successCount + "/" + selectedAngleCount + ")" : "")}
                </h3>
              </div>

              {results.length === 0 && !isGenerating ? (
                <div className="flex flex-col items-center justify-center h-96 text-gray-600">
                  <ImageIcon className="h-16 w-16 mb-4 opacity-30" />
                  <p className="text-gray-400 font-medium">No images generated yet</p>
                  <p className="text-gray-500 text-sm mt-1">Upload an image and click Generate to start</p>
                </div>
              ) : (
                <div className="grid grid-cols-3 gap-3">
                  {getGridItems().map((item) => (
                    <div key={item.name} className="rounded-xl overflow-hidden bg-gray-800 group relative">
                      {item.success && "image_data" in item && item.image_data ? (
                        <>
                          <img
                            src={"data:" + (item.content_type || "image/png") + ";base64," + item.image_data}
                            alt={item.name}
                            className="w-full h-auto aspect-square object-cover"
                          />
                          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                            <button
                              onClick={() => downloadImage(item as AngleResult)}
                              className="bg-white/20 backdrop-blur-sm rounded-lg p-2 hover:bg-white/30 transition-colors"
                            >
                              <Download className="h-5 w-5 text-white" />
                            </button>
                          </div>
                          {"provider" in item && item.provider && (
                            <div className={"absolute top-1.5 left-1.5 px-1.5 py-0.5 rounded text-[10px] font-medium text-white " + getProviderBadgeColor(item.provider)}>
                              {getProviderLabel(item.provider)}
                            </div>
                          )}
                          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/70 to-transparent px-2 py-1.5">
                            <p className="text-white text-xs font-medium">{item.name}</p>
                          </div>
                        </>
                      ) : !isPending(item) && item.error ? (
                        <div className="w-full aspect-square flex flex-col items-center justify-center bg-gray-800 gap-2 p-3">
                          <p className="text-gray-400 text-xs text-center">{item.name}</p>
                          <button
                            onClick={() => retryAngle(item.name)}
                            disabled={retryingAngle === item.name}
                            className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-red-600/80 text-white hover:bg-red-500 transition-colors disabled:opacity-50"
                          >
                            {retryingAngle === item.name ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <RotateCcw className="h-4 w-4" />
                            )}
                            Retry
                          </button>
                        </div>
                      ) : (
                        <div className="w-full aspect-square flex flex-col items-center justify-center gap-2">
                          <Loader2 className="h-6 w-6 animate-spin text-gray-500" />
                          <p className="text-gray-500 text-xs">{item.name}</p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <footer className="border-t border-gray-800/50 mt-12 py-6 text-center">
        <p className="text-gray-500 text-sm">AI NADIR ANGLE</p>
        <p className="text-gray-600 text-xs mt-1">&copy; 2026 Multi-Angle Image Generator</p>
      </footer>
    </div>
  );
}

function AppWithErrorBoundary() {
  return (
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  );
}

export default AppWithErrorBoundary;
