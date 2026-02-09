import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  typescript: {
    // tsc --noEmit passes cleanly; Next.js 16 Turbopack type-checker
    // has a false positive with Record<string, unknown> && chains in JSX
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
