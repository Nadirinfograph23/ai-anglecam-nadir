import { useState, useRef, useCallback, useEffect, Component, type ReactNode, type ErrorInfo } from "react";
import {
  Camera,
  Upload,
  Download,
  Loader2,
  AlertCircle,
  ImageIcon,
  Sparkles,
  X,
  ChevronDown,
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

// ===== Multiple HuggingFace Space URLs for failover =====
const HF_SPACE_URLS = [
  "https://linoyts-qwen-image-edit-angles.hf.space",
  "https://linoyts-qwen2-5-image-edit.hf.space",
];

let currentSpaceIndex = 0;

function getNextSpaceUrl(): string {
  currentSpaceIndex = (currentSpaceIndex + 1) % HF_SPACE_URLS.length;
  return HF_SPACE_URLS[currentSpaceIndex];
}

function getCurrentSpaceUrl(): string {
  return HF_SPACE_URLS[currentSpaceIndex];
}

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
  { name: "Low Angle", h: 0, v: -30 },
  { name: "Bird Eye 45", h: 45, v: 45 },
  { name: "Dutch Angle", h: 30, v: 15 },
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

// ===== HuggingFace Gradio API Client with failover =====
async function uploadToHF(imageBlob: Blob, spaceUrl: string): Promise<string> {
  const formData = new FormData();
  formData.append("files", imageBlob, "input.png");

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 60000);

  try {
    const response = await fetch(spaceUrl + "/gradio_api/upload", {
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
  spaceUrl: string,
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

    const submitResponse = await fetch(spaceUrl + "/gradio_api/call/maybe_infer", {
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
      spaceUrl + "/gradio_api/call/maybe_infer/" + eventId,
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

/** Retry with failover across multiple HF Space endpoints */
async function withFailoverRetry(
  fn: (spaceUrl: string) => Promise<{ imageData: string; contentType: string }>,
  maxRetries = 3,
  baseDelay = 2000,
  label = "",
): Promise<{ imageData: string; contentType: string }> {
  let lastError: Error | null = null;
  const totalAttempts = maxRetries * HF_SPACE_URLS.length;

  for (let attempt = 0; attempt < totalAttempts; attempt++) {
    const spaceUrl = getCurrentSpaceUrl();
    try {
      return await fn(spaceUrl);
    } catch (e) {
      lastError = e instanceof Error ? e : new Error(String(e));
      console.warn(
        "[HF] " + label + " attempt " + (attempt + 1) + "/" + totalAttempts +
        " failed on " + spaceUrl + ":", lastError.message
      );
      getNextSpaceUrl();
      if (attempt < totalAttempts - 1) {
        const delay = Math.min(baseDelay * Math.pow(1.5, attempt % maxRetries), 15000);
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }
  throw lastError || new Error(label + " failed after " + totalAttempts + " attempts");
}

/** Upload with failover across endpoints */
async function uploadWithFailover(imageBlob: Blob): Promise<{ path: string; spaceUrl: string }> {
  let lastError: Error | null = null;
  for (let i = 0; i < HF_SPACE_URLS.length * 2; i++) {
    const spaceUrl = getCurrentSpaceUrl();
    try {
      const path = await uploadToHF(imageBlob, spaceUrl);
      return { path, spaceUrl };
    } catch (e) {
      lastError = e instanceof Error ? e : new Error(String(e));
      console.warn("[HF] Upload failed on " + spaceUrl + ":", lastError.message);
      getNextSpaceUrl();
      if (i < HF_SPACE_URLS.length * 2 - 1) {
        await new Promise((r) => setTimeout(r, 2000));
      }
    }
  }
  throw lastError || new Error("Upload failed on all endpoints");
}

// ===== 3D Camera Preview Component =====
function CameraPreview3D({
  horizontalAngle,
  verticalAngle,
  imageSrc,
}: {
  horizontalAngle: number;
  verticalAngle: number;
  imageSrc: string | null;
}) {
  const rotateY = -horizontalAngle;
  const rotateX = verticalAngle * 0.5;

  return (
    <div
      className="relative w-full aspect-square rounded-2xl overflow-hidden border-2 border-gray-700 bg-gray-900/60"
      style={{ perspective: "800px" }}
    >
      <div className="absolute inset-0 flex items-center justify-center">
        <div
          className="w-[70%] h-[70%] relative"
          style={{
            transformStyle: "preserve-3d",
            transform: "rotateX(" + (20 + rotateX) + "deg) rotateY(" + rotateY + "deg)",
            transition: "transform 0.4s ease-out",
          }}
        >
          {imageSrc ? (
            <div
              className="absolute inset-0 rounded-xl overflow-hidden shadow-2xl"
              style={{
                backfaceVisibility: "hidden",
                transform: "translateZ(1px)",
              }}
            >
              <img src={imageSrc} alt="Preview" className="w-full h-full object-cover" />
              <div className="absolute inset-0 bg-gradient-to-t from-black/20 to-transparent" />
            </div>
          ) : (
            <div
              className="absolute inset-0 rounded-xl border-2 border-dashed border-gray-600 flex items-center justify-center bg-gray-800/50"
              style={{
                backfaceVisibility: "hidden",
                transform: "translateZ(1px)",
              }}
            >
              <ImageIcon className="h-12 w-12 text-gray-600" />
            </div>
          )}

          <div
            className="absolute left-[-20%] right-[-20%] h-[60%] bottom-[-30%]"
            style={{
              transform: "rotateX(90deg) translateZ(-1px)",
              backgroundImage: "linear-gradient(rgba(100, 200, 255, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(100, 200, 255, 0.1) 1px, transparent 1px)",
              backgroundSize: "20% 20%",
            }}
          />

          <div
            className="absolute w-3 h-3 rounded-full bg-cyan-400 shadow-lg shadow-cyan-400/50"
            style={{
              top: "-15%",
              left: "50%",
              transform: "translateX(-50%) translateZ(40px)",
            }}
          />
          <div
            className="absolute w-0.5 h-8 bg-cyan-400/60"
            style={{
              top: "-15%",
              left: "50%",
              transform: "translateX(-50%) translateZ(20px) rotateX(-30deg)",
              transformOrigin: "top center",
            }}
          />
        </div>
      </div>

      <div className="absolute top-3 left-3 bg-gray-900/80 backdrop-blur-sm rounded-lg px-2.5 py-1.5 text-xs font-mono text-cyan-400">
        {"H: " + horizontalAngle + "\u00B0 / V: " + verticalAngle + "\u00B0"}
      </div>

      <div className="absolute bottom-3 right-3 w-12 h-12">
        <svg viewBox="0 0 48 48" className="w-full h-full">
          <circle cx="24" cy="24" r="20" fill="rgba(0,0,0,0.5)" stroke="rgba(100,200,255,0.3)" strokeWidth="1" />
          <text x="24" y="10" textAnchor="middle" fill="rgba(100,200,255,0.6)" fontSize="7" fontWeight="bold">N</text>
          <text x="24" y="42" textAnchor="middle" fill="rgba(100,200,255,0.4)" fontSize="6">S</text>
          <text x="6" y="26" textAnchor="middle" fill="rgba(100,200,255,0.4)" fontSize="6">W</text>
          <text x="42" y="26" textAnchor="middle" fill="rgba(100,200,255,0.4)" fontSize="6">E</text>
          <line
            x1="24" y1="24"
            x2={24 + 14 * Math.sin((horizontalAngle * Math.PI) / 180)}
            y2={24 - 14 * Math.cos((horizontalAngle * Math.PI) / 180)}
            stroke="#22d3ee"
            strokeWidth="2"
            strokeLinecap="round"
          />
          <circle cx="24" cy="24" r="2" fill="#22d3ee" />
        </svg>
      </div>
    </div>
  );
}

// ===== Dropdown Component =====
function AngleDropdown({
  selectedAngle,
  onSelect,
}: {
  selectedAngle: typeof PREDEFINED_ANGLES[0];
  onSelect: (angle: typeof PREDEFINED_ANGLES[0]) => void;
}) {
  const [isOpen, setIsOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div ref={dropdownRef} className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between gap-2 px-4 py-3 rounded-xl bg-gray-800 border border-gray-700 text-white hover:border-cyan-500/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-cyan-500/20 to-blue-600/20 flex items-center justify-center">
            <Camera className="h-4 w-4 text-cyan-400" />
          </div>
          <div className="text-left">
            <div className="text-sm font-medium">{selectedAngle.name}</div>
            <div className="text-xs text-gray-400">
              {"H: " + selectedAngle.h + "\u00B0 / V: " + selectedAngle.v + "\u00B0"}
            </div>
          </div>
        </div>
        <ChevronDown className={"h-4 w-4 text-gray-400 transition-transform " + (isOpen ? "rotate-180" : "")} />
      </button>

      {isOpen && (
        <div className="absolute z-50 w-full mt-2 py-1 rounded-xl bg-gray-800 border border-gray-700 shadow-xl shadow-black/50 max-h-72 overflow-y-auto">
          {PREDEFINED_ANGLES.map((angle) => (
            <button
              key={angle.name}
              onClick={() => { onSelect(angle); setIsOpen(false); }}
              className={"w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors " +
                (selectedAngle.name === angle.name
                  ? "bg-cyan-500/10 text-cyan-400"
                  : "text-gray-300 hover:bg-gray-700/50"
                )}
            >
              <div className={"w-6 h-6 rounded-md flex items-center justify-center text-xs font-bold " +
                (selectedAngle.name === angle.name ? "bg-cyan-500/20 text-cyan-400" : "bg-gray-700 text-gray-400")}>
                {angle.name.charAt(0)}
              </div>
              <div>
                <div className="text-sm font-medium">{angle.name}</div>
                <div className="text-xs text-gray-500">{"H: " + angle.h + "\u00B0 / V: " + angle.v + "\u00B0"}</div>
              </div>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

// ===== Main App =====
function App() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [selectedAngle, setSelectedAngle] = useState(PREDEFINED_ANGLES[0]);
  const [lens, setLens] = useState("normal");
  const [isGenerating, setIsGenerating] = useState(false);
  const [resultImage, setResultImage] = useState<{ imageData: string; contentType: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusMsg, setStatusMsg] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const LENS_OPTIONS = [
    { value: "normal", label: "Normal" },
    { value: "wide", label: "Wide Angle" },
    { value: "closeup", label: "Close-Up" },
  ];

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
    setResultImage(null);
    const reader = new FileReader();
    reader.onload = (e) => setSelectedImage(e.target?.result as string);
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

  const generateAngle = async () => {
    if (!imageFile) return;
    setIsGenerating(true);
    setError(null);
    setResultImage(null);
    setStatusMsg("Optimizing image...");

    try {
      const optimized = await optimizeImage(imageFile);
      setStatusMsg("Uploading to AI server...");

      const { path: uploadedPath, spaceUrl } = await uploadWithFailover(optimized);

      const rotate = clampRotate(selectedAngle.h);
      const forward = convertForward(lens);
      const tilt = convertVertical(selectedAngle.v);
      const isWide = lens === "wide";

      setStatusMsg("Generating " + selectedAngle.name + " view...");

      const result = await withFailoverRetry(
        async (currentUrl) => {
          let finalPath = uploadedPath;
          if (currentUrl !== spaceUrl) {
            setStatusMsg("Re-uploading to backup server...");
            finalPath = await uploadToHF(optimized, currentUrl);
          }
          return generateSingleAngleFromHF(finalPath, rotate, forward, tilt, isWide, currentUrl);
        },
        3, 2000, selectedAngle.name
      );

      setResultImage(result);
      setStatusMsg(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed. Please try again.");
      setStatusMsg(null);
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadImage = () => {
    if (!resultImage) return;
    const ext = resultImage.contentType.includes("webp") ? "webp" : "png";
    const link = document.createElement("a");
    link.href = "data:" + resultImage.contentType + ";base64," + resultImage.imageData;
    link.download = "angle-" + selectedAngle.name.toLowerCase().replace(/\s+/g, "-") + "." + ext;
    link.click();
  };

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
                  AI Camera Angle Generator
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <Sparkles className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Powered by Qwen Image Edit</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white">Generate Camera Angle View</h2>
          <p className="text-gray-400 text-sm mt-1">
            Upload an image, choose a camera angle, and generate a new perspective instantly
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Controls */}
          <div className="space-y-5">
            {/* Upload Section */}
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
                    <img src={selectedImage} alt="Input preview" className="w-full h-auto max-h-48 object-contain" />
                    <button
                      onClick={() => { setSelectedImage(null); setImageFile(null); setResultImage(null); }}
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

            {/* Angle Selection Dropdown */}
            <div className="rounded-2xl bg-gray-900/60 border border-gray-800/50 p-5">
              <h3 className="text-white font-semibold mb-3">Camera Angle</h3>
              <AngleDropdown selectedAngle={selectedAngle} onSelect={setSelectedAngle} />
            </div>

            {/* Lens Type */}
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

            {/* Generate Button */}
            <button
              onClick={generateAngle}
              disabled={!imageFile || isGenerating}
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-blue-600 to-blue-700 px-6 py-3.5 text-base font-semibold text-white shadow-lg shadow-blue-500/20 transition-all hover:shadow-blue-500/30 hover:from-blue-500 hover:to-blue-600 disabled:opacity-40 disabled:cursor-not-allowed"
            >
              {isGenerating ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Sparkles className="h-5 w-5" />
                  Generate Angle View
                </>
              )}
            </button>

            {/* Status */}
            {statusMsg && (
              <div className="rounded-xl bg-blue-900/20 border border-blue-800/30 p-3 flex items-center gap-2">
                <Loader2 className="h-4 w-4 text-blue-400 animate-spin shrink-0" />
                <p className="text-blue-300 text-sm">{statusMsg}</p>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="rounded-xl bg-red-900/30 border border-red-800/50 p-3 flex items-start gap-2">
                <AlertCircle className="h-4 w-4 text-red-400 mt-0.5 shrink-0" />
                <p className="text-red-300 text-sm">{error}</p>
              </div>
            )}
          </div>

          {/* Right Panel - 3D Preview and Result */}
          <div className="lg:col-span-2 space-y-5">
            {/* 3D Camera Preview */}
            <div className="rounded-2xl bg-gray-900/60 border border-gray-800/50 p-5">
              <h3 className="text-white font-semibold mb-3">3D Camera Angle Preview</h3>
              <CameraPreview3D
                horizontalAngle={selectedAngle.h}
                verticalAngle={selectedAngle.v}
                imageSrc={selectedImage}
              />
            </div>

            {/* Generated Result */}
            <div className="rounded-2xl bg-gray-900/60 border border-gray-800/50 p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-white font-semibold">Generated Result</h3>
                {resultImage && (
                  <button
                    onClick={downloadImage}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-gradient-to-r from-blue-600 to-cyan-600 text-white hover:from-blue-500 hover:to-cyan-500 transition-colors"
                  >
                    <Download className="h-3.5 w-3.5" />
                    Download
                  </button>
                )}
              </div>

              {resultImage ? (
                <div className="rounded-xl overflow-hidden bg-gray-800 group relative">
                  <img
                    src={"data:" + resultImage.contentType + ";base64," + resultImage.imageData}
                    alt={"Generated " + selectedAngle.name + " view"}
                    className="w-full h-auto max-h-[600px] object-contain"
                  />
                  <div className="absolute bottom-3 left-3 bg-gray-900/80 backdrop-blur-sm rounded-lg px-3 py-1.5 text-xs text-cyan-400 font-medium">
                    {selectedAngle.name + " (" + selectedAngle.h + "\u00B0, " + selectedAngle.v + "\u00B0)"}
                  </div>
                </div>
              ) : isGenerating ? (
                <div className="flex flex-col items-center justify-center h-64 text-gray-600">
                  <Loader2 className="h-12 w-12 animate-spin text-blue-500 mb-4" />
                  <p className="text-gray-400 font-medium">Generating angle view...</p>
                  <p className="text-gray-500 text-sm mt-1">{statusMsg || "Please wait..."}</p>
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-64 text-gray-600">
                  <ImageIcon className="h-16 w-16 mb-4 opacity-30" />
                  <p className="text-gray-400 font-medium">No image generated yet</p>
                  <p className="text-gray-500 text-sm mt-1">Upload an image and click Generate to start</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      <footer className="border-t border-gray-800/50 mt-12 py-6 text-center">
        <p className="text-gray-500 text-sm">AI NADIR ANGLE</p>
        <p className="text-gray-600 text-xs mt-1">&copy; 2026 AI Camera Angle Generator</p>
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
