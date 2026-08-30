import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",

  distDir: process.env.NEXT_DIST_DIR || ".next",

  allowedDevOrigins: ["127.0.0.1"],

  reactCompiler: true,

  experimental: {
    optimizePackageImports: ["lucide-react", "motion"],
  },
};

export default nextConfig;
