import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  env: {
    // Server-side only — used by /api/livekit/token route
    LIVEKIT_API_KEY: process.env.LIVEKIT_API_KEY ?? "",
    LIVEKIT_API_SECRET: process.env.LIVEKIT_API_SECRET ?? "",
  },
  async rewrites() {
    // In development, proxy /backend/* to avoid CORS issues
    const backendUrl = process.env.API_URL ?? "http://localhost:8000";
    return [{ source: "/backend/:path*", destination: `${backendUrl}/:path*` }];
  },
};

export default nextConfig;
