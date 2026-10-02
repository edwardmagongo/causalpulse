import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  // Stops the dev server writing instruction files of its own into this directory.
  agentRules: false,
};

export default nextConfig;
