import { useState, useRef, useCallback, useEffect } from "react";
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
  Clock,
  Zap,
  Eye,
} from "lucide-react";

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// --- Client-side image compression ---
async function compressImage(file: File, maxWidth = 2048, quality = 0.85): Promise<File> {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      let w = img.width;
      let h = img.height;
      if (w > maxWidth) {
        h = (h * maxWidth) / w;
        w = maxWidth;
      }
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) { resolve(file); return; }
      ctx.drawImage(img, 0, 0, w, h);
      canvas.toBlob(
        (blob) => {
          if (!blob || blob.size >= file.size) { resolve(file); return; }
          resolve(new File([blob], file.name, { type: "image/jpeg" }));
        },
        "image/jpeg",
        quality
      );
    };
    img.onerror = () => resolve(file);
    img.src = URL.createObjectURL(file);
  });
}

// --- LocalStorage cache for results ---
const CACHE_KEY_PREFIX = "anglecam_cache_";
const CACHE_TTL = 60 * 60 * 1000; // 1 hour

function getCachedResults(imageHash: string, lens: string): AngleResult[] | null {
  try {
    const key = CACHE_KEY_PREFIX + imageHash + "_" + lens;
    const raw = localStorage.getItem(key);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (Date.now() - parsed.timestamp > CACHE_TTL) {
      localStorage.removeItem(key);
      return null;
    }
    return parsed.results;
  } catch {
    return null;
  }
}

function setCachedResults(imageHash: string, lens: string, results: AngleResult[]): void {
  try {
    const key = CACHE_KEY_PREFIX + imageHash + "_" + lens;
    const successResults = results.filter((r) => r.success);
    if (successResults.length === 0) return;
    localStorage.setItem(key, JSON.stringify({ results: successResults, timestamp: Date.now() }));
    // Evict old entries if localStorage is getting full
    cleanLocalStorageCache();
  } catch {
    // localStorage full or unavailable
  }
}

function cleanLocalStorageCache(): void {
  try {
    const keys: string[] = [];
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key && key.startsWith(CACHE_KEY_PREFIX)) keys.push(key);
    }
    // Keep at most 5 cached sessions
    if (keys.length > 5) {
      const sorted = keys.map((k) => {
        try {
          const data = JSON.parse(localStorage.getItem(k) || "");
          return { key: k, ts: data.timestamp || 0 };
        } catch {
          return { key: k, ts: 0 };
        }
      }).sort((a, b) => a.ts - b.ts);
      for (let i = 0; i < sorted.length - 5; i++) {
        localStorage.removeItem(sorted[i].key);
      }
    }
  } catch {
    // ignore
  }
}

async function computeFileHash(file: File): Promise<string> {
  const buffer = await file.arrayBuffer();
  const hashBuffer = await crypto.subtle.digest("SHA-256", buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, "0")).join("").slice(0, 16);
}

interface AngleResult {
  name: string;
  success: boolean;
  image_data?: string;
  content_type?: string;
  error?: string;
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

const ANGLE_LABELS: Record<string, string> = {
  Front: "Front View",
  "Front Right": "Front Right",
  Right: "Right Side",
  "Back Right": "Back Right",
  Back: "Back View",
  "Back Left": "Back Left",
  Left: "Left Side",
  "Front Left": "Front Left",
  "Top View": "Top View",
};

const LENS_OPTIONS = [
  { value: "normal", label: "Normal" },
  { value: "wide", label: "Wide Angle" },
  { value: "closeup", label: "Close-Up" },
];

function isPending(item: GridItem): item is PendingAngle {
  return "pending" in item && item.pending === true;
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
  const [isCompressing, setIsCompressing] = useState(false);
  const [startTime, setStartTime] = useState<number | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [imageHash, setImageHash] = useState<string | null>(null);
  const [cachedCount, setCachedCount] = useState(0);
  const [compareAngle, setCompareAngle] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Elapsed time counter
  useEffect(() => {
    if (!startTime || !isGenerating) return;
    const interval = setInterval(() => {
      setElapsed(Math.floor((Date.now() - startTime) / 1000));
    }, 1000);
    return () => clearInterval(interval);
  }, [startTime, isGenerating]);

  const handleFileSelect = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file (JPEG, PNG, WebP)");
      return;
    }
    if (file.size > 20 * 1024 * 1024) {
      setError("Image too large. Maximum size is 20MB.");
      return;
    }

    setError(null);
    setResults([]);
    setCachedCount(0);

    // Compress image client-side before setting
    setIsCompressing(true);
    try {
      const compressed = await compressImage(file);
      setImageFile(compressed);
      const hash = await computeFileHash(compressed);
      setImageHash(hash);

      const reader = new FileReader();
      reader.onload = (e) => setSelectedImage(e.target?.result as string);
      reader.readAsDataURL(compressed);
    } catch {
      setImageFile(file);
      const reader = new FileReader();
      reader.onload = (e) => setSelectedImage(e.target?.result as string);
      reader.readAsDataURL(file);
    } finally {
      setIsCompressing(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleFileSelect(file);
    },
    [handleFileSelect]
  );

  const generateAllAngles = async () => {
    if (!imageFile) return;

    // Check client-side cache first
    if (imageHash) {
      const cached = getCachedResults(imageHash, lens);
      if (cached && cached.length > 0) {
        setResults(cached);
        setCachedCount(cached.length);
        setProgress({ completed: cached.length, total: 9 });
        // If all 9 are cached, skip generation
        if (cached.length === 9) return;
      }
    }

    setIsGenerating(true);
    setError(null);
    if (cachedCount === 0) setResults([]);
    setProgress((prev) => ({ completed: prev.completed, total: 9 }));
    setStartTime(Date.now());
    setElapsed(0);

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("lens", lens);

    const allResults: AngleResult[] = [];

    try {
      const response = await fetch(API_URL + "/api/generate-stream", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.detail || "Server error: " + response.status);
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response stream available");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const chunk of lines) {
          const dataLine = chunk.split("\n").find((l) => l.startsWith("data: "));
          if (!dataLine) continue;

          try {
            const data = JSON.parse(dataLine.substring(6));
            if (data.type === "error") {
              setError(data.message);
            } else if (data.type === "result") {
              const newResult: AngleResult = {
                name: data.name,
                success: data.success,
                image_data: data.image_data,
                content_type: data.content_type,
                error: data.error,
              };
              allResults.push(newResult);
              setResults((prev) => {
                const existing = prev.filter((r) => r.name !== data.name);
                return [...existing, newResult];
              });
              setProgress({ completed: data.completed, total: data.total });
            } else if (data.type === "done") {
              setProgress({ completed: data.completed, total: data.total });
            }
          } catch {
            // skip
          }
        }
      }

      // Cache successful results
      if (imageHash && allResults.length > 0) {
        setCachedResults(imageHash, lens, allResults);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setIsGenerating(false);
      setStartTime(null);
    }
  };

  const retryAngle = async (angleName: string) => {
    if (!imageFile) return;
    setRetryingAngle(angleName);
    setError(null);

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("angle_name", angleName);
    formData.append("lens", lens);

    try {
      const response = await fetch(API_URL + "/api/retry-angle", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.detail || "Retry failed: " + response.status);
      }
      const data = await response.json();
      setResults((prev) => {
        const existing = prev.filter((r) => r.name !== angleName);
        return [...existing, {
          name: data.name,
          success: data.success,
          image_data: data.image_data,
          content_type: data.content_type,
        }];
      });
    } catch (e) {
      setError("Retry for " + angleName + " failed: " + (e instanceof Error ? e.message : "Unknown error"));
    } finally {
      setRetryingAngle(null);
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

  const successCount = results.filter((r) => r.success).length;
  const failCount = results.filter((r) => !r.success).length;

  const getGridItems = (): GridItem[] => {
    if (isGenerating) {
      return ANGLE_NAMES.map((name) => {
        const existing = results.find((r) => r.name === name);
        if (existing) return existing;
        return { name, success: false as const, pending: true as const };
      });
    }
    return ANGLE_NAMES
      .map((name) => results.find((r) => r.name === name))
      .filter((r): r is AngleResult => r !== undefined);
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
                  Multi-Angle Image Generator
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
          <h2 className="text-2xl font-bold text-white">Generate Multi-Angle Views</h2>
          <p className="text-gray-400 text-sm mt-1">
            Upload an image and generate 9 different viewing angles automatically
          </p>
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

            <button
              onClick={generateAllAngles}
              disabled={!imageFile || isGenerating}
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
                  Generate All 9 Angles
                </>
              )}
            </button>

            {isCompressing && (
              <div className="rounded-xl bg-blue-900/30 border border-blue-800/50 p-3 flex items-center gap-2">
                <Zap className="h-4 w-4 text-blue-400 animate-pulse" />
                <p className="text-blue-300 text-sm">Compressing image for faster upload...</p>
              </div>
            )}

            {isGenerating && (
              <div className="rounded-xl bg-gray-800 p-3 space-y-2">
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-gradient-to-r from-blue-500 to-cyan-500 h-2 rounded-full transition-all duration-500"
                    style={{ width: ((progress.completed / progress.total) * 100) + "%" }}
                  />
                </div>
                <div className="flex items-center justify-between text-xs">
                  <p className="text-gray-400">
                    {progress.completed + "/" + progress.total + " angles completed"}
                  </p>
                  <p className="text-gray-500 flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {elapsed + "s elapsed"}
                  </p>
                </div>
                <p className="text-gray-500 text-xs text-center">
                  {"Estimated: ~" + Math.max(10, (9 - progress.completed) * 15) + "s remaining"}
                </p>
              </div>
            )}

            {cachedCount > 0 && !isGenerating && (
              <div className="rounded-xl bg-cyan-900/20 border border-cyan-800/30 p-3 flex items-center gap-2">
                <Zap className="h-4 w-4 text-cyan-400" />
                <p className="text-cyan-300 text-sm">{cachedCount + " results loaded from local cache"}</p>
              </div>
            )}

            {error && (
              <div className="rounded-xl bg-red-900/30 border border-red-800/50 p-3 flex items-start gap-2">
                <AlertCircle className="h-4 w-4 text-red-400 mt-0.5 shrink-0" />
                <p className="text-red-300 text-sm">{error}</p>
              </div>
            )}

            {results.length > 0 && !isGenerating && (
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
            )}
          </div>

          <div className="lg:col-span-2">
            <div className="rounded-2xl bg-gray-900/60 border border-gray-800/50 p-5">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-white font-semibold">
                  {"Generated Angles" + (results.length > 0 ? " (" + successCount + "/9)" : "")}
                </h3>
              </div>

              {results.length === 0 && !isGenerating ? (
                <div className="flex flex-col items-center justify-center h-96 text-gray-600">
                  <ImageIcon className="h-16 w-16 mb-4 opacity-30" />
                  <p className="text-gray-400 font-medium">No images generated yet</p>
                  <p className="text-gray-500 text-sm mt-1">Upload an image and click Generate to start</p>
                  <div className="mt-6 text-left w-full max-w-xs space-y-2">
                    <p className="text-gray-400 text-xs font-semibold uppercase tracking-wider">How it works</p>
                    <div className="flex items-start gap-2">
                      <span className="text-cyan-400 text-xs font-bold mt-0.5">1</span>
                      <p className="text-gray-500 text-xs">Upload any image (product, object, etc.)</p>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-cyan-400 text-xs font-bold mt-0.5">2</span>
                      <p className="text-gray-500 text-xs">Choose a lens type (normal, wide, close-up)</p>
                    </div>
                    <div className="flex items-start gap-2">
                      <span className="text-cyan-400 text-xs font-bold mt-0.5">3</span>
                      <p className="text-gray-500 text-xs">AI generates 9 different angle views automatically</p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-3 gap-3">
                  {getGridItems().map((item) => (
                    <div key={item.name} className="rounded-xl overflow-hidden bg-gray-800 group relative">
                      {item.success && "image_data" in item && item.image_data ? (
                        <>
                          <img
                            src={"data:" + (item.content_type || "image/webp") + ";base64," + item.image_data}
                            alt={item.name}
                            className="w-full h-auto aspect-square object-cover"
                          />
                          <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-2">
                            <button
                              onClick={() => downloadImage(item as AngleResult)}
                              className="bg-white/20 backdrop-blur-sm rounded-lg p-2 hover:bg-white/30 transition-colors"
                              title="Download"
                            >
                              <Download className="h-5 w-5 text-white" />
                            </button>
                            {selectedImage && (
                              <button
                                onClick={() => setCompareAngle(compareAngle === item.name ? null : item.name)}
                                className="bg-white/20 backdrop-blur-sm rounded-lg p-2 hover:bg-white/30 transition-colors"
                                title="Compare with original"
                              >
                                <Eye className="h-5 w-5 text-white" />
                              </button>
                            )}
                          </div>
                        </>
                      ) : !isPending(item) && item.error ? (
                        <div className="w-full aspect-square flex flex-col items-center justify-center gap-2 p-3">
                          <AlertCircle className="h-6 w-6 text-red-400" />
                          <p className="text-red-400 text-xs text-center truncate w-full">
                            {item.error || "Failed"}
                          </p>
                          <button
                            onClick={() => retryAngle(item.name)}
                            disabled={retryingAngle === item.name}
                            className="flex items-center gap-1 px-2 py-1 rounded text-xs bg-red-900/50 text-red-300 hover:bg-red-800/50 transition-colors disabled:opacity-50"
                          >
                            {retryingAngle === item.name ? (
                              <Loader2 className="h-3 w-3 animate-spin" />
                            ) : (
                              <RotateCcw className="h-3 w-3" />
                            )}
                            Retry
                          </button>
                        </div>
                      ) : (
                        <div className="w-full aspect-square flex items-center justify-center">
                          <Loader2 className="h-6 w-6 animate-spin text-gray-500" />
                        </div>
                      )}
                      <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/80 to-transparent p-2">
                        <p className="text-white text-xs font-medium text-center">
                          {ANGLE_LABELS[item.name] || item.name}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>

      {/* Before/After comparison modal */}
      {compareAngle && selectedImage && (
        <div className="fixed inset-0 z-50 bg-black/80 flex items-center justify-center p-4" onClick={() => setCompareAngle(null)}>
          <div className="bg-gray-900 rounded-2xl p-4 max-w-4xl w-full" onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-white font-semibold">{'Before / After - ' + compareAngle}</h3>
              <button onClick={() => setCompareAngle(null)} className="p-1 rounded-full hover:bg-gray-700">
                <X className="h-5 w-5 text-gray-400" />
              </button>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-gray-400 text-xs mb-2 text-center">Original</p>
                <img src={selectedImage} alt="Original" className="w-full rounded-lg object-contain max-h-96" />
              </div>
              <div>
                <p className="text-gray-400 text-xs mb-2 text-center">{compareAngle}</p>
                {(() => {
                  const r = results.find((r) => r.name === compareAngle);
                  return r && r.image_data ? (
                    <img
                      src={"data:" + (r.content_type || "image/webp") + ";base64," + r.image_data}
                      alt={compareAngle}
                      className="w-full rounded-lg object-contain max-h-96"
                    />
                  ) : (
                    <div className="w-full aspect-square bg-gray-800 rounded-lg flex items-center justify-center">
                      <p className="text-gray-500">Not available</p>
                    </div>
                  );
                })()}
              </div>
            </div>
          </div>
        </div>
      )}

      <footer className="border-t border-gray-800/50 mt-12 py-6 text-center">
        <p className="text-gray-500 text-sm">AI NADIR ANGLE</p>
        <p className="text-gray-600 text-xs mt-1">&copy; 2026 Multi-Angle Image Generator</p>
      </footer>
    </div>
  );
}

export default App;
