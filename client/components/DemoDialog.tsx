import { Dialog, DialogContent, DialogTitle, DialogTrigger } from "@/components/ui/dialog";

export default function DemoButton() {
  const demoSrc = import.meta.env.VITE_DEMO_VIDEO_URL || "https://cdn.coverr.co/videos/coverr-night-drive-timelapse-3988/1080p.mp4";
  return (
    <Dialog>
      <DialogTrigger className="rounded-full border border-white/10 bg-white/5 px-5 py-2 text-sm text-white/80 transition hover:bg-white/10">Watch Demo</DialogTrigger>
      <DialogContent className="max-w-3xl">
        <DialogTitle className="text-white">Website Demo</DialogTitle>
        <video className="mt-2 w-full rounded-lg" src={demoSrc} controls autoPlay />
      </DialogContent>
    </Dialog>
  );
}
