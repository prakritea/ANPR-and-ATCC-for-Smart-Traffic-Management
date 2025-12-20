import { useEffect, useMemo, useState } from "react";
import { Button } from "@/components/ui/button";

interface Article {
  title: string;
  url: string;
  description?: string;
  source?: { name?: string };
  publishedAt?: string;
  urlToImage?: string;
}

export default function News() {
  const [articles, setArticles] = useState<Article[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const key = import.meta.env.VITE_NEWS_API_KEY as string | undefined;

  const fetchNews = async () => {
    setLoading(true);
    setError(null);
    try {
      if (!key) throw new Error("Missing VITE_NEWS_API_KEY. Showing demo data.");
      const url = `https://newsapi.org/v2/everything?q=india%20traffic%20road%20transport%20highway&language=en&sortBy=publishedAt&pageSize=12&apiKey=${key}`;
      const res = await fetch(url);
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      setArticles(data.articles || []);
    } catch (e: any) {
      setError(e?.message || "Failed to load news");
      setArticles(demoArticles);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchNews(); }, []);

  const items = useMemo(() => articles.slice(0, 12), [articles]);

  return (
    <main className="min-h-screen px-6 py-24">
      <div className="mx-auto max-w-6xl">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-3xl font-bold text-white">Traffic News</h1>
          <Button onClick={fetchNews} className="rounded-full" variant="secondary">Refresh</Button>
        </div>
        {error && <div className="mb-4 text-xs text-white/60">{error}</div>}
        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
          {items.map((a, i) => (
            <article key={i} className="overflow-hidden rounded-2xl border border-white/10 bg-white/5 p-4 text-white">
              {a.urlToImage && (
                // eslint-disable-next-line @next/next/no-img-element
                <img src={a.urlToImage} alt="" className="mb-3 aspect-video w-full rounded-lg object-cover" />
              )}
              <h3 className="line-clamp-2 text-lg font-semibold">{a.title}</h3>
              <p className="mt-2 line-clamp-3 text-sm text-white/70">{a.description}</p>
              <div className="mt-3 flex items-center justify-between text-xs text-white/60">
                <span>{a.source?.name}</span>
                <span>{a.publishedAt ? new Date(a.publishedAt).toLocaleString() : ""}</span>
              </div>
              <a href={a.url} target="_blank" rel="noreferrer" className="mt-3 inline-block text-sm text-primary underline">Read more</a>
            </article>
          ))}
          {items.length === 0 && (
            <div className="col-span-full text-white/60">No news available.</div>
          )}
        </div>
      </div>
    </main>
  );
}

const demoArticles: Article[] = [
  {
    title: "Delhi traffic advisory issued amid construction work across key corridors",
    url: "https://example.com/delhi-traffic",
    description: "Authorities advise commuters to plan journeys as maintenance affects peak-hour flow.",
    source: { name: "Demo" },
    publishedAt: new Date().toISOString(),
  },
  {
    title: "NH-48 upgrade: Temporary diversions announced for heavy vehicles",
    url: "https://example.com/nh48",
    description: "Logistics operators alerted; follow posted signs to avoid congestion.",
    source: { name: "Demo" },
    publishedAt: new Date().toISOString(),
  },
  {
    title: "Mumbai introduces smart signals to ease junction bottlenecks",
    url: "https://example.com/mumbai-smart-signals",
    description: "Adaptive signal control to reduce average wait time by 20% in pilot phase.",
    source: { name: "Demo" },
    publishedAt: new Date().toISOString(),
  },
];
