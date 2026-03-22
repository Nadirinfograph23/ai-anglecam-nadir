import { useState, useRef, useCallback } from "react";
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

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

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
  const [retryingAll, setRetryingAll] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

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

  const processStream = async (response: Response, onResult?: (data: AngleResult) => void) => {
    const reader = response.body?.getReader();
    if (!reader) throw new Error("No response stream available");

    const decoder = new TextDecoder();
    let buffer = "";
    const streamResults: AngleResult[] = [];

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
            const result: AngleResult = {
              name: data.name,
              success: data.success,
              image_data: data.image_data,
              content_type: data.content_type,
              error: data.error,
            };
            streamResults.push(result);
            setResults((prev) => {
              const existing = prev.filter((r) => r.name !== data.name);
              return [...existing, result];
            });
            setProgress({ completed: data.completed, total: data.total });
            if (onResult) onResult(result);
          } else if (data.type === "done") {
            setProgress({ completed: data.completed, total: data.total });
          }
        } catch {
          // skip
        }
      }
    }
    return streamResults;
  };

  const generateAllAngles = async () => {
    if (!imageFile) return;
    setIsGenerating(true);
    setError(null);
    setResults([]);
    setProgress({ completed: 0, total: 9 });

    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("lens", lens);

    try {
      const response = await fetch(API_URL + "/api/generate-stream", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(errData?.detail || "Server error: " + response.status);
      }

      const streamResults = await processStream(response);

      // Auto-retry failed angles once
      const failedAngles = streamResults.filter((r) => !r.success);
      if (failedAngles.length > 0 && failedAngles.length < 9) {
        for (const failed of failedAngles) {
          try {
            const retryFormData = new FormData();
            retryFormData.append("image", imageFile);
            retryFormData.append("angle_name", failed.name);
            retryFormData.append("lens", lens);

            const retryResponse = await fetch(API_URL + "/api/retry-angle", {
              method: "POST",
              body: retryFormData,
            });
            if (retryResponse.ok) {
              const data = await retryResponse.json();
              if (data.success) {
                setResults((prev) => {
                  const existing = prev.filter((r) => r.name !== data.name);
                  return [...existing, {
                    name: data.name,
                    success: data.success,
                    image_data: data.image_data,
                    content_type: data.content_type,
                  }];
                });
              }
            }
          } catch {
            // silent retry failure
          }
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed");
    } finally {
      setIsGenerating(false);
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

  const retryAllFailed = async () => {
    if (!imageFile) return;
    const failedResults = results.filter((r) => !r.success);
    if (failedResults.length === 0) return;
    setRetryingAll(true);
    setError(null);

    for (const failed of failedResults) {
      try {
        const formData = new FormData();
        formData.append("image", imageFile);
        formData.append("angle_name", failed.name);
        formData.append("lens", lens);

        const response = await fetch(API_URL + "/api/retry-angle", {
          method: "POST",
          body: formData,
        });
        if (response.ok) {
          const data = await response.json();
          setResults((prev) => {
            const existing = prev.filter((r) => r.name !== data.name);
            return [...existing, {
              name: data.name,
              success: data.success,
              image_data: data.image_data,
              content_type: data.content_type,
            }];
          });
        }
      } catch {
        // continue with next angle
      }
    }
    setRetryingAll(false);
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
    // Only show successful results - hide failed ones
    return ANGLE_NAMES
      .map((name) => results.find((r) => r.name === name))
      .filter((r): r is AngleResult => r !== undefined && r.success);
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
                      {"تم توليد " + successCount + " صور من أصل " + (successCount + failCount) + " بسبب الضغط على الخادم. عاود المحاولة بعد قليل."}
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
                  {"Generated Angles" + (results.length > 0 ? " (" + successCount + "/9)" : "")}
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
                            src={"data:" + (item.content_type || "image/webp") + ";base64," + item.image_data}
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

      <footer className="border-t border-gray-800/50 mt-12 py-6 text-center">
        <p className="text-gray-500 text-sm">AI NADIR ANGLE</p>
        <p className="text-gray-600 text-xs mt-1">&copy; 2026 Multi-Angle Image Generator</p>
      </footer>
    </div>
  );
}

export default App;
