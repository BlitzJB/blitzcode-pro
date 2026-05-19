import type { Metadata } from "next";
import { Geist, Geist_Mono, Instrument_Serif } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

const instrumentSerif = Instrument_Serif({
  variable: "--font-serif",
  subsets: ["latin"],
  weight: "400",
  style: ["normal", "italic"],
});

export const metadata: Metadata = {
  title: "blitzcode",
  description: "Agent workspace for the LLM project.",
};

// Runs before React hydrates. Reads localStorage.theme (the FOUC mirror
// of server settings.appearance.theme), falls back to the OS preference,
// and stamps data-theme on <html>. Without this the page would paint
// light first, then flip — visible flash on every reload for dark users.
// Runs before React hydrates. Does three jobs in one script (kept inline
// so it executes before any paint or fetch):
//
//   1. Resolve the saved theme preference and stamp data-theme on <html>
//      so we never flash light when the user prefers dark.
//   2. Stamp data-tauri="1" when inside the Tauri shell so CSS variants
//      can branch on host context.
//   3. **LAN auth bootstrap**: if the URL carries `?k=<token>` (the
//      QR-code shape), save it to localStorage and strip from the URL
//      so the token doesn't pollute history / Referer headers. Then
//      monkeypatch `fetch` so every same-origin request automatically
//      gets `Authorization: Bearer <token>`. Loopback hosts (Tauri
//      WebView) won't have a saved token and won't need one — the
//      server bypasses auth for them.
const themeBootScript = `(function(){try{
var p=localStorage.getItem('blitz.theme');
var r=p==='dark'||p==='light'?p:(window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');
document.documentElement.setAttribute('data-theme',r);
if(window.__TAURI_INTERNALS__)document.documentElement.setAttribute('data-tauri','1');
var u=new URL(window.location.href);
var k=u.searchParams.get('k');
if(k){localStorage.setItem('blitz.lan.token',k);u.searchParams.delete('k');window.history.replaceState({},'',u.toString());}
var tok=localStorage.getItem('blitz.lan.token');
if(tok){
var orig=window.fetch.bind(window);
window.fetch=function(input,init){
init=init||{};
try{
var url=typeof input==='string'?input:(input&&input.url)||'';
var same=url.startsWith('/')||url.startsWith(window.location.origin);
if(same){
var h=new Headers(init.headers||(typeof input!=='string'&&input.headers)||{});
if(!h.has('authorization'))h.set('authorization','Bearer '+tok);
init.headers=h;
}
}catch(e){}
return orig(input,init);
};
}
}catch(e){}})();`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} ${instrumentSerif.variable} h-full antialiased`}
      suppressHydrationWarning
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeBootScript }} />
      </head>
      <body className="min-h-full flex flex-col">{children}</body>
    </html>
  );
}
