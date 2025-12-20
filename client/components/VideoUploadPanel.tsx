import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Slider } from "@/components/ui/slider";
import { toast } from "sonner";
import { processVideo } from "@/services/processingService";

export default function VideoUploadPanel() {
  const [file, setFile] = useState<File | null>(null);
  const [model, setModel] = useState("yolo11l.pt");
  const [useEasyOCR, setUseEasyOCR] = useState(true);
  const [useGPU, setUseGPU] = useState(false);
  const [attempts, setAttempts] = useState<number>(3);
  const [minBoxArea, setMinBoxArea] = useState<number>(2000);
  const [lineY, setLineY] = useState<number>(430);
  const [outputCsv, setOutputCsv] = useState<string>("vehicle_log.csv");
  const [loading, setLoading] = useState(false);

  const onSubmit = async () => {
    if (!file) {
      toast.error("Please select an MP4 file");
      return;
    }
    const form = new FormData();
    form.append("file", file);
    form.append("yolo_weights", model);
    form.append("use_easyocr", String(useEasyOCR));
    form.append("use_gpu", String(useGPU));
    form.append("max_ocr_attempts", String(attempts));
    form.append("min_box_area", String(minBoxArea));
    form.append("line_y", String(lineY));
    form.append("output_csv", outputCsv);

    setLoading(true);
    const id = toast.loading("Processing video...");
    try {
      const res = await processVideo(form);
      toast.success(res.message || "Processing complete", { id });
    } catch (e: any) {
      toast.error(e?.message || "Failed to process video", { id });
    } finally {
      setLoading(false);
    }
  };

  return (
    <section id="process" className="relative">
      <div className="mx-auto max-w-4xl rounded-2xl border border-white/10 bg-white/5 p-6 shadow-2xl backdrop-blur">
        <h2 className="mb-1 text-2xl font-semibold text-white">Video Processing</h2>
        <p className="mb-6 text-sm text-white/60">Upload an MP4 and set parameters to match your Python backend.</p>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
          <div className="space-y-4">
            <label className="block text-sm text-white/80">Video (MP4)</label>
            <Input
              type="file"
              accept="video/mp4"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />

            <label className="block text-sm text-white/80">YOLO Weights</label>
            <Select value={model} onValueChange={setModel}>
              <SelectTrigger>
                <SelectValue placeholder="Select model" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="yolo11l.pt">yolo11l.pt</SelectItem>
              </SelectContent>
            </Select>

            <div className="flex items-center justify-between pt-2">
              <div>
                <p className="text-sm font-medium text-white/90">Use EasyOCR</p>
                <p className="text-xs text-white/60">Enable plate recognition</p>
              </div>
              <Switch checked={useEasyOCR} onCheckedChange={setUseEasyOCR} />
            </div>

            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-white/90">Use GPU</p>
                <p className="text-xs text-white/60">Enable EasyOCR GPU if available</p>
              </div>
              <Switch checked={useGPU} onCheckedChange={setUseGPU} />
            </div>

            <div>
              <label className="block text-sm text-white/80">Output CSV Filename</label>
              <Input value={outputCsv} onChange={(e) => setOutputCsv(e.target.value)} />
            </div>
          </div>

          <div className="space-y-6">
            <div>
              <div className="mb-2 flex items-end justify-between">
                <div>
                  <p className="text-sm font-medium text-white/90">Max OCR Attempts</p>
                  <p className="text-xs text-white/60">Retries per vehicle</p>
                </div>
                <span className="text-sm text-white/70">{attempts}</span>
              </div>
              <Slider
                value={[attempts]}
                onValueChange={(v) => setAttempts(v[0] ?? 1)}
                min={1}
                max={3}
                step={1}
              />
            </div>

            <div>
              <div className="mb-2 flex items-end justify-between">
                <div>
                  <p className="text-sm font-medium text-white/90">Min Box Area (px²)</p>
                  <p className="text-xs text-white/60">Filter small detections</p>
                </div>
                <span className="text-sm text-white/70">{minBoxArea}</span>
              </div>
              <Slider
                value={[minBoxArea]}
                onValueChange={(v) => setMinBoxArea(v[0] ?? 0)}
                min={0}
                max={20000}
                step={500}
              />
            </div>

            <div>
              <div className="mb-2 flex items-end justify-between">
                <div>
                  <p className="text-sm font-medium text-white/90">Counting Line Y (px)</p>
                  <p className="text-xs text-white/60">Horizontal line position</p>
                </div>
                <span className="text-sm text-white/70">{lineY}</span>
              </div>
              <Slider
                value={[lineY]}
                onValueChange={(v) => setLineY(v[0] ?? 0)}
                min={0}
                max={1080}
                step={10}
              />
            </div>

            <div className="pt-2">
              <Button onClick={onSubmit} disabled={loading} className="w-full rounded-full">
                {loading ? "Processing..." : "Start Processing"}
              </Button>
              <p className="mt-2 text-xs text-white/50">Note: The Python backend processes and writes to CSV server-side.</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
