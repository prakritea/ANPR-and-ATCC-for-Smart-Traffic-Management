import useEmblaCarousel from "embla-carousel-react";
import { useCallback, useEffect } from "react";

const FACTS: { title: string; text: string }[] = [
  { title: "Seat Belts", text: "Wearing seat belts is mandatory for front passengers across India." },
  { title: "Speed Limits", text: "National highways typically have 80–100 km/h limits for cars; obey posted signs." },
  { title: "No Honking Zones", text: "Hospitals and schools often mark silent zones; avoid honking." },
  { title: "Helmet Rule", text: "IS-certified helmets are compulsory for riders and pillion on two-wheelers." },
  { title: "Lane Discipline", text: "Keep left except to overtake; use indicators and avoid sudden lane changes." },
  { title: "Drink & Drive", text: "Blood alcohol limit is 30 mg/100 ml; strict penalties and license suspension." },
  { title: "Mobile Use", text: "Using a handheld phone while driving is prohibited; use hands-free only when safe." },
  { title: "Zebra Crossings", text: "Always yield to pedestrians on marked crossings." },
  { title: "Emergency Vehicles", text: "Give way to ambulances and fire brigades; keep left and stop if needed." },
  { title: "Child Safety", text: "Use appropriate child restraint seats; never seat children in the front." },
];

export default function FactsCarousel() {
  const [emblaRef, emblaApi] = useEmblaCarousel({ loop: true, align: "start" });

  const autoplay = useCallback(() => {
    if (!emblaApi) return;
    const timer = setInterval(() => {
      if (!emblaApi) return;
      if (emblaApi.canScrollNext()) emblaApi.scrollNext();
      else emblaApi.scrollTo(0);
    }, 3000);
    return () => clearInterval(timer);
  }, [emblaApi]);

  useEffect(() => {
    const cleanup = autoplay();
    return () => cleanup && cleanup();
  }, [autoplay]);

  return (
    <div className="relative" id="facts">
      <div className="pointer-events-none absolute inset-0 rounded-2xl bg-gradient-to-b from-white/5 via-transparent to-transparent" />
      <div className="overflow-hidden" ref={emblaRef}>
        <div className="flex gap-6">
          {FACTS.map((f, i) => (
            <div
              key={i}
              className="min-w-[85%] md:min-w-[48%] lg:min-w-[32%]"
            >
              <article className="h-full rounded-2xl border border-white/10 bg-white/5 p-6 shadow-inner">
                <h3 className="mb-2 text-lg font-semibold text-white/90">{f.title}</h3>
                <p className="text-sm text-white/70">{f.text}</p>
              </article>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
