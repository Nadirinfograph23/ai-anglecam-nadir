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
              className="px-4 py-2 bg-[#CDFF00] text-gray-900 rounded-lg hover:bg-[#d8ff33] transition-colors"
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

// ===== Camera Angle Presets with professional terminology =====
const PREDEFINED_ANGLES = [
  { name: "Eye Level", h: 0, v: 0 },
  { name: "3/4 Front Right", h: 45, v: 0 },
  { name: "Profile Right", h: 90, v: 0 },
  { name: "3/4 Back Right", h: 135, v: 0 },
  { name: "Rear View", h: 180, v: 0 },
  { name: "3/4 Back Left", h: -135, v: 0 },
  { name: "Profile Left", h: -90, v: 0 },
  { name: "3/4 Front Left", h: -45, v: 0 },
  { name: "Bird's Eye View", h: 0, v: 60 },
  { name: "Low Angle", h: 0, v: -30 },
  { name: "High Angle 3/4", h: 45, v: 45 },
  { name: "Dutch Angle", h: 30, v: 15 },
];

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

// ===== Image Optimization (returns base64 string for API route) =====
async function optimizeImage(file: File, maxSize = 2048): Promise<string> {
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
        (blob) => {
          if (!blob) { reject(new Error("Image conversion failed")); return; }
          const reader = new FileReader();
          reader.onload = () => {
            const result = reader.result as string;
            resolve(result.split(",")[1] || "");
          };
          reader.onerror = reject;
          reader.readAsDataURL(blob);
        },
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

// ===== API call through Vercel serverless function =====
// ===== Provider Options =====
const PROVIDER_OPTIONS = [
  { value: "", label: "Auto (All Providers)", description: "Fastest available" },
  { value: "anglechanger", label: "AngleChanger.ai", description: "Dedicated angle AI" },
  { value: "hf-space", label: "HuggingFace Space", description: "Qwen Image Edit" },
  { value: "hf-inference", label: "HF Inference", description: "Zero-1-to-3" },
  { value: "replicate", label: "Replicate", description: "Cloud GPU" },
  { value: "stable-horde", label: "Stable Horde", description: "Distributed" },
];

async function generateAngleViaAPI(
  imageBase64: string,
  rotateDeg: number,
  moveForward: number,
  verticalTilt: number,
  wideangle: boolean,
  provider?: string,
): Promise<{ imageData: string; contentType: string }> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 300_000); // 5 min max

  try {
    const body: Record<string, unknown> = {
      imageData: imageBase64,
      rotateDeg,
      moveForward,
      verticalTilt,
      wideangle,
    };
    if (provider) {
      body.provider = provider;
    }

    const response = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({ error: "Server error" })) as { error?: string };
      throw new Error(errData.error || "Server error (" + response.status + ")");
    }

    const data = await response.json() as {
      success: boolean;
      imageData?: string;
      contentType?: string;
      error?: string;
      provider?: string;
    };

    if (!data.success || !data.imageData) {
      throw new Error(data.error || "Generation failed");
    }

    console.log("[Generate] Success via " + data.provider);
    return { imageData: data.imageData, contentType: data.contentType || "image/png" };
  } finally {
    clearTimeout(timeout);
  }
}

// ===== Interactive 3D Camera Preview Component =====
function CameraPreview3D({
  horizontalAngle,
  verticalAngle,
  imageSrc,
  onAngleChange,
}: {
  horizontalAngle: number;
  verticalAngle: number;
  imageSrc: string | null;
  onAngleChange?: (h: number, v: number) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);
  const lastPos = useRef({ x: 0, y: 0 });

  const rotateY = -horizontalAngle;
  const rotateX = verticalAngle * 0.5;

  const handlePointerDown = useCallback((e: React.PointerEvent) => {
    if (!onAngleChange) return;
    isDragging.current = true;
    lastPos.current = { x: e.clientX, y: e.clientY };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, [onAngleChange]);

  const handlePointerMove = useCallback((e: React.PointerEvent) => {
    if (!isDragging.current || !onAngleChange) return;
    const dx = e.clientX - lastPos.current.x;
    const dy = e.clientY - lastPos.current.y;
    lastPos.current = { x: e.clientX, y: e.clientY };

    const sensitivity = 0.5;
    let newH = horizontalAngle + dx * sensitivity;
    let newV = verticalAngle - dy * sensitivity;

    newH = clampRotate(newH);
    newV = Math.max(-60, Math.min(60, newV));

    newH = Math.round(newH / 5) * 5;
    newV = Math.round(newV / 5) * 5;

    onAngleChange(newH, newV);
  }, [horizontalAngle, verticalAngle, onAngleChange]);

  const handlePointerUp = useCallback(() => {
    isDragging.current = false;
  }, []);

  const handleCompassClick = useCallback((e: React.MouseEvent<SVGSVGElement>) => {
    if (!onAngleChange) return;
    const svg = e.currentTarget;
    const rect = svg.getBoundingClientRect();
    const cx = rect.left + rect.width / 2;
    const cy = rect.top + rect.height / 2;
    const dx = e.clientX - cx;
    const dy = e.clientY - cy;
    const angle = Math.atan2(dx, -dy) * (180 / Math.PI);
    const snapped = Math.round(angle / 15) * 15;
    onAngleChange(clampRotate(snapped), verticalAngle);
  }, [onAngleChange, verticalAngle]);

  return (
    <div
      ref={containerRef}
      className={"relative w-full aspect-square rounded-2xl overflow-hidden border-2 bg-black/60 " +
        (onAngleChange ? "border-[#CDFF00]/30 cursor-grab active:cursor-grabbing" : "border-gray-700")}
      style={{ perspective: "800px" }}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerLeave={handlePointerUp}
    >
      <div className="absolute inset-0 flex items-center justify-center">
        <div
          className="w-[70%] h-[70%] relative"
          style={{
            transformStyle: "preserve-3d",
            transform: "rotateX(" + (20 + rotateX) + "deg) rotateY(" + rotateY + "deg)",
            transition: isDragging.current ? "none" : "transform 0.4s ease-out",
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
              <img src={imageSrc} alt="Preview" className="w-full h-full object-cover pointer-events-none" />
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
              backgroundImage: "linear-gradient(rgba(205, 255, 0, 0.1) 1px, transparent 1px), linear-gradient(90deg, rgba(205, 255, 0, 0.1) 1px, transparent 1px)",
              backgroundSize: "20% 20%",
            }}
          />

          <div
            className="absolute w-3 h-3 rounded-full bg-[#CDFF00] shadow-lg shadow-[#CDFF00]/50"
            style={{
              top: "-15%",
              left: "50%",
              transform: "translateX(-50%) translateZ(40px)",
            }}
          />
          <div
            className="absolute w-0.5 h-8 bg-[#CDFF00]/60"
            style={{
              top: "-15%",
              left: "50%",
              transform: "translateX(-50%) translateZ(20px) rotateX(-30deg)",
              transformOrigin: "top center",
            }}
          />
        </div>
      </div>

      <div className="absolute top-3 left-3 bg-black/80 backdrop-blur-sm rounded-lg px-2.5 py-1.5 text-xs font-mono text-[#CDFF00]">
        {"H: " + horizontalAngle + "\u00B0 / V: " + verticalAngle + "\u00B0"}
      </div>

      {onAngleChange && (
        <div className="absolute top-3 right-3 bg-gray-900/80 backdrop-blur-sm rounded-lg px-2.5 py-1.5 text-xs text-gray-400">
          Drag to rotate
        </div>
      )}

      <div className="absolute bottom-3 right-3 w-14 h-14">
        <svg viewBox="0 0 48 48" className="w-full h-full cursor-pointer" onClick={handleCompassClick}>
          <circle cx="24" cy="24" r="20" fill="rgba(0,0,0,0.5)" stroke="rgba(205,255,0,0.3)" strokeWidth="1" />
          <text x="24" y="10" textAnchor="middle" fill="rgba(205,255,0,0.6)" fontSize="7" fontWeight="bold">N</text>
          <text x="24" y="42" textAnchor="middle" fill="rgba(205,255,0,0.4)" fontSize="6">S</text>
          <text x="6" y="26" textAnchor="middle" fill="rgba(205,255,0,0.4)" fontSize="6">W</text>
          <text x="42" y="26" textAnchor="middle" fill="rgba(205,255,0,0.4)" fontSize="6">E</text>
          <line
            x1="24" y1="24"
            x2={24 + 14 * Math.sin((horizontalAngle * Math.PI) / 180)}
            y2={24 - 14 * Math.cos((horizontalAngle * Math.PI) / 180)}
            stroke="#CDFF00"
            strokeWidth="2"
            strokeLinecap="round"
          />
          <circle cx="24" cy="24" r="2" fill="#CDFF00" />
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
        className="w-full flex items-center justify-between gap-2 px-4 py-3 rounded-xl bg-gray-900 border border-gray-700 text-white hover:border-[#CDFF00]/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#CDFF00]/10 flex items-center justify-center">
            <Camera className="h-4 w-4 text-[#CDFF00]" />
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
                  ? "bg-[#CDFF00]/10 text-[#CDFF00]"
                  : "text-gray-300 hover:bg-gray-700/50"
                )}
            >
              <div className={"w-6 h-6 rounded-md flex items-center justify-center text-xs font-bold " +
                (selectedAngle.name === angle.name ? "bg-[#CDFF00]/20 text-[#CDFF00]" : "bg-gray-700 text-gray-400")}>
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
  const [provider, setProvider] = useState("");
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

  // Handle interactive angle change from 3D preview drag
  const handleAngleChange = useCallback((h: number, v: number) => {
    let closest = PREDEFINED_ANGLES[0];
    let minDist = Infinity;
    for (const angle of PREDEFINED_ANGLES) {
      const dist = Math.abs(angle.h - h) + Math.abs(angle.v - v);
      if (dist < minDist) {
        minDist = dist;
        closest = angle;
      }
    }
    if (minDist <= 15) {
      setSelectedAngle(closest);
    } else {
      setSelectedAngle({ name: "Custom (" + h + "\u00B0, " + v + "\u00B0)", h, v });
    }
  }, []);

  const generateAngle = async () => {
    if (!imageFile) return;
    setIsGenerating(true);
    setError(null);
    setResultImage(null);
    setStatusMsg("Optimizing image...");

    try {
      const imageBase64 = await optimizeImage(imageFile);
      setStatusMsg("Generating " + selectedAngle.name + " view...");

      const rotate = clampRotate(selectedAngle.h);
      const forward = convertForward(lens);
      const tilt = convertVertical(selectedAngle.v);
      const isWide = lens === "wide";

      const result = await generateAngleViaAPI(
        imageBase64,
        rotate,
        forward,
        tilt,
        isWide,
        provider || undefined,
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
    <div className="min-h-screen bg-black text-white">
      <header className="sticky top-0 z-30 w-full border-b border-gray-800/50 bg-black/90 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6">
          <div className="flex items-center justify-between h-14">
            <div className="flex items-center gap-3">
              <div className="rounded-lg bg-[#CDFF00] p-1.5">
                <Camera className="h-5 w-5 text-black" />
              </div>
              <div>
                <h1 className="text-base font-bold text-[#CDFF00]">
                  AI AngleCam Nadir
                </h1>
                <p className="text-xs text-gray-500 -mt-0.5 hidden sm:block">
                  AI Camera Angle Generator
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 text-xs text-gray-500">
              <Sparkles className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Multi-Provider AI Engine</span>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 py-6">
        <div className="mb-6">
          <h2 className="text-2xl font-bold text-white">Generate Camera Angle View</h2>
          <p className="text-gray-400 text-sm mt-1">
            Upload an image, choose a camera angle or drag to set a custom angle, and generate a new perspective
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Panel - Controls */}
          <div className="space-y-5">
            {/* Upload Section */}
            <div className="rounded-2xl bg-gray-950/60 border border-gray-800/50 p-5">
              <div className="flex items-center gap-2 mb-3">
                <h3 className="text-white font-semibold">Input Image</h3>
                <span className="text-red-500">*</span>
              </div>
              <div onDrop={handleDrop} onDragOver={(e) => e.preventDefault()} className="space-y-3">
                <button
                  onClick={() => fileInputRef.current?.click()}
                  className={"w-full flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed p-6 transition-all " + (
                    selectedImage
                      ? "border-gray-700 bg-gray-900/30 hover:border-[#CDFF00]/50"
                      : "border-gray-600 bg-gray-900/50 hover:border-[#CDFF00]/50 hover:bg-gray-900"
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
            <div className="rounded-2xl bg-gray-950/60 border border-gray-800/50 p-5">
              <h3 className="text-white font-semibold mb-3">Camera Angle</h3>
              <AngleDropdown selectedAngle={selectedAngle} onSelect={setSelectedAngle} />
              <p className="text-xs text-gray-500 mt-2">
                Or drag on the 3D preview to set a custom angle
              </p>
            </div>

            {/* Lens Type */}
            <div className="rounded-2xl bg-gray-950/60 border border-gray-800/50 p-5">
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

            {/* Provider Selection */}
            <div className="rounded-2xl bg-gray-950/60 border border-gray-800/50 p-5">
              <h3 className="text-white font-semibold mb-3">AI Provider</h3>
              <div className="space-y-2">
                {PROVIDER_OPTIONS.map((opt) => (
                  <button
                    key={opt.value}
                    onClick={() => setProvider(opt.value)}
                    className={"w-full flex items-center justify-between px-4 py-2.5 rounded-xl text-left transition-all " + (
                      provider === opt.value
                        ? "bg-[#CDFF00]/10 border border-[#CDFF00]/40 text-white"
                        : "bg-gray-900 border border-gray-700 text-gray-300 hover:border-gray-500"
                    )}
                  >
                    <span className="text-sm font-medium">{opt.label}</span>
                    <span className={"text-xs " + (provider === opt.value ? "text-[#CDFF00]/70" : "text-gray-500")}>
                      {opt.description}
                    </span>
                  </button>
                ))}
              </div>
            </div>

            {/* Generate Button */}
            <button
              onClick={generateAngle}
              disabled={!imageFile || isGenerating}
              className="w-full flex items-center justify-center gap-2 rounded-xl bg-[#CDFF00] px-6 py-3.5 text-base font-semibold text-black shadow-lg shadow-[#CDFF00]/20 transition-all hover:bg-[#d8ff33] hover:shadow-[#CDFF00]/30 disabled:opacity-40 disabled:cursor-not-allowed"
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
              <div className="rounded-xl bg-[#CDFF00]/5 border border-[#CDFF00]/20 p-3 flex items-center gap-2">
                <Loader2 className="h-4 w-4 text-[#CDFF00] animate-spin shrink-0" />
                <p className="text-[#CDFF00]/80 text-sm">{statusMsg}</p>
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
            <div className="rounded-2xl bg-gray-950/60 border border-gray-800/50 p-5">
              <h3 className="text-white font-semibold mb-3">3D Camera Angle Preview</h3>
              <CameraPreview3D
                horizontalAngle={selectedAngle.h}
                verticalAngle={selectedAngle.v}
                imageSrc={selectedImage}
                onAngleChange={handleAngleChange}
              />
            </div>

            {/* Generated Result */}
            <div className="rounded-2xl bg-gray-950/60 border border-gray-800/50 p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-white font-semibold">Generated Result</h3>
                {resultImage && (
                  <button
                    onClick={downloadImage}
                    className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-[#CDFF00] text-black hover:bg-[#d8ff33] transition-colors"
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
                  <div className="absolute bottom-3 left-3 bg-black/80 backdrop-blur-sm rounded-lg px-3 py-1.5 text-xs text-[#CDFF00] font-medium">
                    {selectedAngle.name + " (" + selectedAngle.h + "\u00B0, " + selectedAngle.v + "\u00B0)"}
                  </div>
                </div>
              ) : isGenerating ? (
                <div className="flex flex-col items-center justify-center h-64 text-gray-600">
                  <Loader2 className="h-12 w-12 animate-spin text-[#CDFF00] mb-4" />
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
