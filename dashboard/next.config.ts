import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    const backendUrl = process.env.API_URL ?? "http://localhost:8000";
    return [{ source: "/backend/:path*", destination: `${backendUrl}/:path*` }];
  },
};

export default nextConfig;
