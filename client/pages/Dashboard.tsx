import { useEffect, useMemo, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { toast } from "sonner";
import { uploadVideo, startProcessing, fetchResults } from "@/services/processingService";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip as ReTooltip,
  XAxis,
  YAxis,
  Legend,
} from "recharts";

interface DetectionRow {
  id: number | string;
  type: string;
  plate: string;
  timestamp: string;
}

export default function Dashboard() {
  const [file, setFile] = useState<File | null>(null);
  const [videoURL, setVideoURL] = useState<string | null>(null);
  const [processedURL, setProcessedURL] = useState<string | null>(null);
  const [processing, setProcessing] = useState(false);
  const [rows, setRows] = useState<DetectionRow[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({ Car: 0, Motorcycle: 0, Bus: 0, Truck: 0 });
  const [demo, setDemo] = useState(false);
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!videoURL && file) setVideoURL(URL.createObjectURL(file));
  }, [file, videoURL]);

  useEffect(() => {
    const t = setInterval(async () => {
      if (demo) {
        // generate synthetic logs
        const types = ["Car", "Motorcycle", "Bus", "Truck"] as const;
        const tname = types[Math.floor(Math.random() * types.length)];
        const id = Math.floor(Math.random() * 1000);
        const plate = Math.random().toString(36).slice(2, 6).toUpperCase() + Math.floor(1000 + Math.random() * 9000);
        const row = { id, type: tname, plate, timestamp: new Date().toLocaleTimeString() };
        setRows((r) => [row, ...r].slice(0, 200));
        setCounts((c) => ({ ...c, [tname]: (c[tname] || 0) + 1 }));
        return;
      }
      try {
        const data = await fetchResults();
        if (!data) return;
        if (Array.isArray(data.rows)) setRows(data.rows);
        if (data.counts) setCounts(data.counts);
        if (data.processed_url) setProcessedURL(data.processed_url);
      } catch {}
    }, 1500);
    return () => clearInterval(t);
  }, [demo]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [rows]);

  const total = useMemo(() => Object.values(counts).reduce((a, b) => a + (b || 0), 0), [counts]);

  const distributionData = useMemo(() => (
    [
      { type: "Cars", count: counts.Car || 0 },
      { type: "Motorcycles", count: counts.Motorcycle || 0 },
      { type: "Trucks", count: counts.Truck || 0 },
      { type: "Buses", count: counts.Bus || 0 },
    ]
  ), [counts]);

  const hourlyData = useMemo(() => {
    const map = new Map<string, { hour: string; Car: number; Motorcycle: number; Bus: number; Truck: number; Total: number }>();
    const getHour = (ts: string) => {
      const m = ts.match(/^(\d{1,2})[:.]/);
      if (m) return `${m[1].padStart(2, "0")}:00`;
      try {
        const d = new Date(ts);
        if (!isNaN(d.getTime())) return `${String(d.getHours()).padStart(2, "0")}:00`;
      } catch {}
      return "00:00";
    };
    for (const r of rows) {
      const h = getHour(r.timestamp);
      if (!map.has(h)) map.set(h, { hour: h, Car: 0, Motorcycle: 0, Bus: 0, Truck: 0, Total: 0 });
      const o = map.get(h)!;
      (o as any)[r.type] = ((o as any)[r.type] || 0) + 1;
      o.Total += 1;
    }
    const arr = Array.from(map.values()).sort((a, b) => a.hour.localeCompare(b.hour));
    return arr.slice(-8);
  }, [rows]);

  const onUpload = async () => {
    if (!file) {
      toast.error("Select an MP4 file first");
      return;
    }
    const id = toast.loading("Uploading video...");
    try {
      const { video_url } = await uploadVideo(file);
      if (video_url) setVideoURL(video_url);
      toast.success("Upload complete", { id });
    } catch (e: any) {
      toast.error(e?.message || "Upload failed", { id });
    }
  };

  const onProcess = async () => {
    if (!file) {
      toast.error("Select an MP4 file first");
      return;
    }
    setProcessing(true);
    const id = toast.loading("Starting processing...");
    try {
      const res = await startProcessing();
      if (res?.processed_url) setProcessedURL(res.processed_url);
      toast.success(res?.message || "Processing started", { id });
    } catch (e: any) {
      toast.error(e?.message || "Processing failed", { id });
    } finally {
      setProcessing(false);
    }
  };

  return (
    <main className="min-h-screen px-6 py-24">
      <div className="mx-auto max-w-6xl space-y-6">
        <h1 className="bg-gradient-to-b from-white to-white/85 bg-clip-text text-4xl font-extrabold text-transparent md:text-5xl">Smart Traffic Management – Dashboard</h1>
        <p className="text-white/60">ANPR + ATCC analytics in real-time</p>

        {/* Upload card */}
        <div className="rounded-2xl border border-white/10 bg-white/5 p-4">
          <div className="grid gap-4 md:grid-cols-[1fr_auto_auto] md:items-end">
            <div>
              <label className="mb-2 block text-sm text-white/80">Upload Video (.mp4)</label>
              <div className="rounded-xl border border-dashed border-white/20 bg-white/5 p-6 text-center text-white/70">
                <Input type="file" accept="video/mp4" onChange={(e) => setFile(e.target.files?.[0] || null)} />
                <p className="mt-2 text-xs">Max ~500MB. Supported: MP4</p>
              </div>
            </div>
            <Button onClick={onUpload} className="rounded-full">Upload</Button>
            <Button onClick={onProcess} disabled={processing} className="rounded-full" variant="secondary">
              {processing ? "Processing..." : "Start Processing"}
            </Button>
          </div>
          <div className="mt-3">
            <Button onClick={() => setDemo((d) => !d)} className="rounded-full" variant="ghost">
              {demo ? "Stop Demo" : "Start Demo"}
            </Button>
          </div>
        </div>

        {/* Videos */}
        <div className="grid gap-6 md:grid-cols-2">
          <div className="overflow-hidden rounded-2xl border border-white/10 bg-white/5 p-3 text-white shadow">
            <h2 className="mb-2 text-sm font-semibold">Uploaded Video</h2>
            {videoURL ? (
              <video className="aspect-video w-full rounded-lg" src={videoURL} controls />
            ) : (
              <div className="aspect-video grid place-items-center rounded-lg bg-black/80 text-white/60">No video</div>
            )}
          </div>
          <div className="overflow-hidden rounded-2xl border border-white/10 bg-white/5 p-3 text-white shadow">
            <h2 className="mb-2 text-sm font-semibold">Processed Output</h2>
            {processedURL ? (
              <video className="aspect-video w-full rounded-lg" src={processedURL} controls />
            ) : (
              <div className="aspect-video grid place-items-center rounded-lg bg-black/80 text-white/60">Waiting for output</div>
            )}
          </div>
        </div>

        {/* Analytics panels */}
        <div className="grid gap-6 md:grid-cols-4">
          <StatCard label="Total Vehicles" value={total} />
          <StatCard label="Cars" value={counts.Car || 0} />
          <StatCard label="Motorcycles" value={counts.Motorcycle || 0} />
          <StatCard label="Heavy Vehicles" value={(counts.Bus || 0) + (counts.Truck || 0)} />
        </div>

        <div className="grid gap-6 md:grid-cols-3">
          <div className="md:col-span-2 overflow-hidden rounded-2xl border border-white/10 bg-white/5 p-4 text-white shadow">
            <h3 className="mb-3 text-sm font-semibold">Detections</h3>
            <div className="max-h-[420px] overflow-auto rounded-lg border border-white/10">
              <table className="w-full text-sm">
                <thead className="bg-white/5 text-white/80">
                  <tr>
                    <th className="px-3 py-2 text-left">ID</th>
                    <th className="px-3 py-2 text-left">Vehicle Type</th>
                    <th className="px-3 py-2 text-left">Plate Number</th>
                    <th className="px-3 py-2 text-left">Timestamp</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r, i) => (
                    <tr key={i} className="odd:bg-transparent even:bg-white/5">
                      <td className="px-3 py-2">{r.id}</td>
                      <td className="px-3 py-2">{r.type}</td>
                      <td className="px-3 py-2 font-mono">{r.plate}</td>
                      <td className="px-3 py-2">{r.timestamp}</td>
                    </tr>
                  ))}
                  {rows.length === 0 && (
                    <tr>
                      <td className="px-3 py-6 text-center text-white/60" colSpan={4}>No results yet</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
          <div className="overflow-hidden rounded-2xl border border-white/10 bg-white/5 p-4 text-white shadow">
            <h3 className="mb-3 text-sm font-semibold">Live Log</h3>
            <div ref={logRef} className="max-h-[420px] space-y-2 overflow-auto rounded-lg border border-white/10 p-3 text-xs">
              {rows.slice(0, 50).map((r, i) => (
                <div key={i} className="flex items-center justify-between">
                  <span className="font-medium">#{r.id}</span>
                  <span>{r.type}</span>
                  <span className="font-mono">{r.plate}</span>
                  <span className="text-white/60">{r.timestamp}</span>
                </div>
              ))}
              {rows.length === 0 && <div className="text-white/60">Waiting for detections...</div>}
            </div>
          </div>
        </div>

        {/* Charts */}
        <div className="grid gap-6 md:grid-cols-2">
          <div className="overflow-hidden rounded-2xl border border-white/10 bg-white/5 p-4 text-white shadow">
            <h3 className="mb-3 text-sm font-semibold">Hourly Traffic Flow</h3>
            <div className="h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={hourlyData} margin={{ left: 8, right: 8, bottom: 8 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="hour" stroke="#aaa" />
                  <YAxis stroke="#aaa" />
                  <ReTooltip contentStyle={{ background: "#0b0b0b", border: "1px solid rgba(255,255,255,0.1)", color: "#fff" }} />
                  <Legend />
                  <Line type="monotone" dataKey="Car" stroke="#60a5fa" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="Motorcycle" stroke="#34d399" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="Bus" stroke="#f59e0b" dot={false} strokeWidth={2} />
                  <Line type="monotone" dataKey="Truck" stroke="#f472b6" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
          <div className="overflow-hidden rounded-2xl border border-white/10 bg-white/5 p-4 text-white shadow">
            <h3 className="mb-3 text-sm font-semibold">Vehicle Distribution</h3>
            <div className="h-[280px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={distributionData} margin={{ left: 8, right: 8, bottom: 8 }}>
                  <CartesianGrid stroke="rgba(255,255,255,0.06)" />
                  <XAxis dataKey="type" stroke="#aaa" />
                  <YAxis stroke="#aaa" />
                  <ReTooltip contentStyle={{ background: "#0b0b0b", border: "1px solid rgba(255,255,255,0.1)", color: "#fff" }} />
                  <Bar dataKey="count" fill="#60a5fa" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="overflow-hidden rounded-2xl border border-white/10 bg-white/5 p-4 text-white shadow">
      <div className="text-xs text-white/60">{label}</div>
      <div className="mt-1 text-2xl font-semibold">{value.toLocaleString()}</div>
    </div>
  );
}
