import Hyperspeed from "@/components/Hyperspeed";
import { lazy, Suspense } from "react";
import DemoButton from "@/components/DemoDialog";

const TrafficMap = lazy(() => import("@/components/TrafficMap"));
const FactsCarousel = lazy(() => import("@/components/FactsCarousel"));
const FAQSection = lazy(() => import("@/components/FAQ"));

export default function Index() {
  return (
    <main className="min-h-screen w-full" id="home">
      {/* HERO */}
      <section className="relative flex min-h-[78vh] items-center justify-center overflow-hidden pt-20 md:pt-28 lg:pt-32">
        <Hyperspeed />
        <div className="relative z-10 mx-auto max-w-3xl px-6 text-center">
          <h1 className="bg-gradient-to-b from-white to-white/90 bg-clip-text text-6xl font-extrabold tracking-tight leading-[1.08] text-transparent md:text-8xl pb-2 drop-shadow-[0_8px_28px_rgba(0,0,0,0.55)]">
            Smart Traffic Management
          </h1>
          <p className="mx-auto mt-4 max-w-2xl text-white/70 md:text-lg">
            Upload traffic footage, detect vehicles and plates with YOLO + EasyOCR, and visualize analytics in real-time.
          </p>
          <div className="mt-8 flex items-center justify-center gap-3">
            <a href="/dashboard" className="rounded-full border border-white/10 bg-white/10 px-5 py-2 text-sm font-medium text-white hover:bg-white/20">Get Started</a>
            <DemoButton />
          </div>
        </div>
      </section>

      {/* INFO SECTIONS */}
      <section id="about" className="relative mx-auto mt-16 md:mt-24 mb-16 max-w-6xl px-6">
        <div className="grid gap-6 md:grid-cols-3">
          <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
            <h3 className="text-lg font-semibold text-white">About</h3>
            <p className="mt-2 text-sm text-white/70">A computer vision system for ANPR and traffic analytics. It detects vehicles, reads plates, and aggregates counts in real-time.</p>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
            <h3 className="text-lg font-semibold text-white">System Capabilities</h3>
            <ul className="mt-2 list-disc space-y-1 pl-5 text-sm text-white/70">
              <li>ANPR with EasyOCR</li>
              <li>Vehicle type detection via YOLOv8</li>
              <li>Live overlays, tracking IDs, line crossing</li>
              <li>CSV logging and dashboard analytics</li>
            </ul>
          </div>
          <div className="rounded-2xl border border-white/10 bg-white/5 p-6">
            <h3 className="text-lg font-semibold text-white">How it Works</h3>
            <ol className="mt-2 list-decimal space-y-1 pl-5 text-sm text-white/70">
              <li>Upload video</li>
              <li>YOLO detection + tracking</li>
              <li>EasyOCR plate recognition</li>
              <li>CSV logging</li>
              <li>Dashboard visualization</li>
            </ol>
          </div>
        </div>
      </section>

      {/* LIVE TRAFFIC MAP */}
      <section className="relative mx-auto mt-0 mb-16 max-w-6xl px-6">
        <div className="mb-4 flex items-end justify-between">
          <h2 className="text-2xl font-semibold text-white">Live Traffic Near You</h2>
          <span className="text-xs text-white/60">Uses geolocation and traffic tiles</span>
        </div>
        <Suspense fallback={<div className="h-[420px] w-full animate-pulse rounded-2xl bg-white/5" />}>
          <TrafficMap />
        </Suspense>
      </section>

      {/* FACTS CAROUSEL */}
      <section className="relative mx-auto mb-20 max-w-6xl px-6">
        <div className="mb-4 flex items-end justify-between">
          <h2 className="text-2xl font-semibold text-white">Indian Road Facts & Rules</h2>
          <span className="text-xs text-white/60">Auto-rotating</span>
        </div>
        <Suspense fallback={<div className="h-40 w-full animate-pulse rounded-2xl bg-white/5" />}>
          <FactsCarousel />
        </Suspense>
      </section>

      <Suspense fallback={<div className="h-40 w-full animate-pulse rounded-2xl bg-white/5" />}>
        <FAQSection />
      </Suspense>

    </main>
  );
}
