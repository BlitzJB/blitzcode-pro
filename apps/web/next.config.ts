import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Bundled-app mode: emit a fully static export to `out/`. FastAPI in
  // apps/server mounts that directory at `/`, so frontend + API live on
  // one ephemeral port chosen by the Tauri shell. No Node runtime at
  // user-side launch.
  output: "export",
  // Image optimizer requires a server runtime; disable for static export.
  images: { unoptimized: true },
  // Trailing slashes match the way FastAPI's StaticFiles(html=True)
  // resolves `/foo` → `/foo/index.html`.
  trailingSlash: true,
};

export default nextConfig;
