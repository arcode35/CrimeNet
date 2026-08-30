import type { Metadata } from "next";
import { LandingPage } from "@/components/landing/landing-page";

export const metadata: Metadata = {
  title: "CrimeSense — Spatiotemporal Crime Intelligence",
  description:
    "CrimeSense combines municipal event data with spatial, temporal, environmental, infrastructural, and socioeconomic context to model reported crime intensity.",
};

export default function HomePage() {
  return <LandingPage />;
}
