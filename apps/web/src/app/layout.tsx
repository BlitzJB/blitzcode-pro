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
const themeBootScript = `(function(){try{var p=localStorage.getItem('blitz.theme');var r=p==='dark'||p==='light'?p:(window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light');document.documentElement.setAttribute('data-theme',r);if(window.__TAURI_INTERNALS__)document.documentElement.setAttribute('data-tauri','1');}catch(e){}})();`;

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
