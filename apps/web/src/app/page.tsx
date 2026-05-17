import Chat from "./chat";

// In dev we run the API on a non-standard port (51820) and the Next dev
// server on 51821, so fetches need an absolute base. In the bundled
// .app, FastAPI serves both the API and this page from the same
// ephemeral port — set NEXT_PUBLIC_AGENT_BASE_URL="" at build time so
// the same code uses relative URLs and the origin resolves to whatever
// port the Rust shell happened to pick.
const DEFAULT_BASE_URL = "http://127.0.0.1:51820";

export default function Home() {
  const env = process.env.NEXT_PUBLIC_AGENT_BASE_URL;
  const baseUrl = env === undefined ? DEFAULT_BASE_URL : env;
  return <Chat baseUrl={baseUrl} />;
}
