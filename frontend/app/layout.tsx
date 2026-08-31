import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "maplibre-gl/dist/maplibre-gl.css";
import "./globals.css";
import favicon from "@/assets/logo_icon.png";
import { Providers } from "@/components/providers";

const sans = Geist({ variable: "--font-sans", subsets: ["latin"] });
const mono = Geist_Mono({ variable: "--font-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "CrimeSense — Spatiotemporal Crime Intelligence",
  description:
    "CrimeSense is an interactive geospatial application for exploring reported crime intensity across space and time.",
  applicationName: "CrimeSense",
  icons: {
    icon: [{ url: favicon.src, type: "image/png" }],
    shortcut: [{ url: favicon.src, type: "image/png" }],
    apple: [{ url: favicon.src, type: "image/png" }],
  },
  openGraph: {
    title: "CrimeSense — Spatiotemporal Crime Intelligence",
    description:
      "A data-to-model geospatial platform for reported crime intensity across space and time.",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "CrimeSense — Spatiotemporal Crime Intelligence",
    description:
      "A data-to-model geospatial platform for reported crime intensity across space and time.",
  },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${sans.variable} ${mono.variable}`}>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
