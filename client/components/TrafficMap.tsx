import { useEffect, useRef, useState } from "react";

declare global {
  interface Window {
    maplibregl: any;
  }
}

function loadScript(src: string) {
  return new Promise<void>((resolve, reject) => {
    const existing = document.querySelector(`script[src="${src}"]`);
    if (existing) return resolve();
    const s = document.createElement("script");
    s.src = src;
    s.async = true;
    s.onload = () => resolve();
    s.onerror = () => reject(new Error("Failed to load script"));
    document.head.appendChild(s);
  });
}

function loadCss(href: string) {
  if (document.querySelector(`link[href="${href}"]`)) return;
  const l = document.createElement("link");
  l.rel = "stylesheet";
  l.href = href;
  document.head.appendChild(l);
}

export default function TrafficMap() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let map: any;
    let canceled = false;

    const init = async (center: [number, number]) => {
      try {
        loadCss("https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.css");
        await loadScript("https://unpkg.com/maplibre-gl@3.6.2/dist/maplibre-gl.js");
        if (canceled) return;
        const maptilerKey = import.meta.env.VITE_MAPTILER_KEY as string | undefined;
        const style = maptilerKey
          ? `https://api.maptiler.com/maps/streets-v2-dark/style.json?key=${maptilerKey}`
          : {
              version: 8,
              sources: {
                osm: {
                  type: "raster",
                  tiles: [
                    "https://a.tile.openstreetmap.org/{z}/{x}/{y}.png",
                    "https://b.tile.openstreetmap.org/{z}/{x}/{y}.png",
                    "https://c.tile.openstreetmap.org/{z}/{x}/{y}.png",
                  ],
                  tileSize: 256,
                  attribution:
                    "© OpenStreetMap contributors",
                },
              },
              layers: [
                { id: "osm", type: "raster", source: "osm" },
              ],
              glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
            } as any;

        map = new window.maplibregl.Map({
          container: containerRef.current!,
          style,
          center,
          zoom: 15,
        });
        map.addControl(new window.maplibregl.NavigationControl({ showZoom: true }), "top-right");

        const el = document.createElement("div");
        el.style.width = "14px";
        el.style.height = "14px";
        el.style.borderRadius = "9999px";
        el.style.background = "#60a5fa";
        el.style.boxShadow = "0 0 0 6px rgba(96,165,250,0.35)";
        new window.maplibregl.Marker({ element: el }).setLngLat(center).addTo(map);

        const tomtomKey = import.meta.env.VITE_TOMTOM_KEY as string | undefined;
        if (tomtomKey) {
          map.on("load", () => {
            map.addSource("traffic", {
              type: "raster",
              tiles: [
                `https://api.tomtom.com/traffic/map/4/tile/flow/relative/{z}/{x}/{y}.png?key=${tomtomKey}`,
              ],
              tileSize: 256,
            });
            map.addLayer({ id: "traffic", type: "raster", source: "traffic", paint: { "raster-opacity": 0.75 } });
          });
        } else {
          setError((prev) => prev ? prev : "Traffic overlay requires VITE_TOMTOM_KEY. Showing base map.");
        }
      } catch (e: any) {
        setError(e?.message || "Failed to initialize map");
      }
    };

    const onLocate = () => {
      if (!navigator.geolocation) {
        setError("Geolocation not supported");
        init([77.209, 28.614]); // Delhi fallback (lng, lat)
        return;
      }
      navigator.geolocation.getCurrentPosition(
        (pos) => init([pos.coords.longitude, pos.coords.latitude]),
        () => init([77.209, 28.614]),
        { enableHighAccuracy: true, timeout: 8000 }
      );
    };

    onLocate();

    return () => {
      canceled = true;
      try { map && map.remove && map.remove(); } catch {}
    };
  }, []);

  return (
    <div className="relative">
      <div ref={containerRef} className="h-[420px] w-full rounded-2xl border border-white/10 overflow-hidden" />
      {error && (
        <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-black/70 px-2 py-1 text-xs text-white/80">
          {error}
        </div>
      )}
    </div>
  );
}
