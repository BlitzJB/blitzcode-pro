import Chat from "./chat";

export default function Home() {
  const baseUrl = process.env.NEXT_PUBLIC_AGENT_BASE_URL ?? "http://127.0.0.1:8000";
  return <Chat baseUrl={baseUrl} />;
}
