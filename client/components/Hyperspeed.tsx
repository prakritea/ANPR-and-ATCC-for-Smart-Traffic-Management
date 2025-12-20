import { useEffect, useRef, useState } from "react";

interface Star {
  x: number;
  y: number;
  z: number;
  pz: number;
}

export default function Hyperspeed() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [isBoost, setIsBoost] = useState(false);
  const animationRef = useRef<number | null>(null);
  const starsRef = useRef<Star[]>([]);

  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;

    let width = (canvas.width = canvas.offsetWidth);
    let height = (canvas.height = canvas.offsetHeight);
    let centerX = width / 2;
    let centerY = height / 2;

    const STAR_COUNT = Math.min(900, Math.floor((width * height) / 2000));

    const initStars = () => {
      starsRef.current = Array.from({ length: STAR_COUNT }, () => ({
        x: (Math.random() * 2 - 1) * width,
        y: (Math.random() * 2 - 1) * height,
        z: Math.random() * width,
        pz: 0,
      }));
    };

    initStars();

    const onResize = () => {
      width = canvas.width = canvas.offsetWidth;
      height = canvas.height = canvas.offsetHeight;
      centerX = width / 2;
      centerY = height / 2;
      initStars();
    };

    const observer = new ResizeObserver(onResize);
    observer.observe(canvas);

    const render = () => {
      const ctx2 = ctx;
      const speed = isBoost ? 36 : 10;

      ctx2.fillStyle = "#000";
      ctx2.fillRect(0, 0, width, height);

      for (let s of starsRef.current) {
        s.z -= speed;
        if (s.z < 1) {
          s.x = (Math.random() * 2 - 1) * width;
          s.y = (Math.random() * 2 - 1) * height;
          s.z = width;
          s.pz = s.z;
        }

        const sx = (s.x / s.z) * centerX + centerX;
        const sy = (s.y / s.z) * centerY + centerY;

        const px = (s.x / s.pz) * centerX + centerX;
        const py = (s.y / s.pz) * centerY + centerY;
        s.pz = s.z;

        const alpha = Math.max(0.1, 1 - s.z / width);
        ctx2.strokeStyle = `hsla(220, 90%, 60%, ${alpha})`;
        ctx2.lineWidth = Math.max(1, (1 - s.z / width) * 3);
        ctx2.beginPath();
        ctx2.moveTo(px, py);
        ctx2.lineTo(sx, sy);
        ctx2.stroke();
      }

      animationRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      if (animationRef.current) cancelAnimationFrame(animationRef.current);
      observer.disconnect();
    };
  }, [isBoost]);

  return (
    <div className="absolute inset-0" onPointerDown={() => setIsBoost(true)} onPointerUp={() => setIsBoost(false)} onPointerLeave={() => setIsBoost(false)}>
      <canvas ref={canvasRef} className="h-full w-full block" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.06)_0%,rgba(0,0,0,0.8)_60%,rgba(0,0,0,1)_100%)]" />
    </div>
  );
}
