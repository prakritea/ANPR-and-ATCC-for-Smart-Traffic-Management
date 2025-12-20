import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";

const links = [
  { href: "/", label: "Home" },
  { href: "/dashboard", label: "Dashboard" },
  { href: "/news", label: "News" },
  { href: "/login", label: "Login" },
];

export default function Header() {
  const [scrolled, setScrolled] = useState(false);
  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 12);
    onScroll();
    window.addEventListener("scroll", onScroll);
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <div className="fixed top-4 left-1/2 z-50 -translate-x-1/2">
      <nav
        className={cn(
          "flex items-center gap-1 rounded-full border px-3 py-2 shadow-lg backdrop-blur",
          "border-white/10 bg-white/10",
          scrolled ? "bg-white/10" : "bg-white/5",
        )}
      >
        {links.map((l) => (
          <a
            key={l.href}
            href={l.href}
            className="rounded-full px-3 py-1.5 text-sm text-foreground/80 transition hover:bg-white/10 hover:text-foreground"
          >
            {l.label}
          </a>
        ))}
      </nav>
    </div>
  );
}
