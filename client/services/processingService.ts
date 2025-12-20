const BASE = import.meta.env.VITE_PY_API_URL || "";

export async function processVideo(form: FormData): Promise<{ message: string }>{
  const endpoint = BASE + "/process";
  const res = await fetch(endpoint, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text().catch(() => `Request failed ${res.status}`));
  return res.json();
}

export async function uploadVideo(file: File): Promise<{ video_url?: string }>{
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(BASE + "/upload", { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text().catch(() => `Upload failed ${res.status}`));
  return res.json();
}

export async function startProcessing(): Promise<{ message?: string; processed_url?: string }>{
  const res = await fetch(BASE + "/process", { method: "POST" });
  if (!res.ok) throw new Error(await res.text().catch(() => `Start failed ${res.status}`));
  return res.json();
}

export async function fetchResults(): Promise<any>{
  const res = await fetch(BASE + "/results");
  if (!res.ok) return null;
  return res.json();
}
