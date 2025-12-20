import { useEffect, useState } from "react";

export default function FloatingClock() {
  const [now, setNow] = useState<string>(new Date().toLocaleString());
  useEffect(() => {
    const t = setInterval(() => setNow(new Date().toLocaleString()), 1000);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="fixed bottom-4 right-4 z-50 select-none rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-white/80 shadow-lg backdrop-blur">
      {now}
    </div>
  );
}
