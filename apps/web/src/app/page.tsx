"use client";

import { useEffect, useState } from "react";
import Chat from "./chat";

export default function Home() {
  const [baseUrl, setBaseUrl] = useState<string | null>(null);

  useEffect(() => {
    const env = process.env.NEXT_PUBLIC_AGENT_BASE_URL;
    if (env !== undefined) {
      setBaseUrl(env);
    } else {
      // In bundled .app: FastAPI serves both API and this page from same port.
      // Use window.location.origin for both. In dev: this runs on 51821 but API
      // on ephemeral port, so use relative URLs ("") which will use the current origin.
      setBaseUrl("");
    }
  }, []);

  if (baseUrl === null) return null;
  return <Chat baseUrl={baseUrl} />;
}
