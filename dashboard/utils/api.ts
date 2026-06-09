/**
 * API base URL — reads from env var set in dashboard/.env.local
 * Default: localhost:8000 (works for local dev without any config)
 */
export const API_BASE =
  (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000") + "/api";

export async function fetcher<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`[${res.status}] ${body || res.statusText}`);
  }
  return res.json() as Promise<T>;
}
