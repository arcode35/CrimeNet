import type { Metadata } from "next";
import { CrimeExplorer } from "@/components/explorer/crime-explorer";

export const metadata: Metadata = {
  title: "Explorer — CrimeNet",
  description: "Explore CrimeNet's H3 spatiotemporal intensity and inference-coverage contract.",
};

export default function ExplorerPage() {
  return <CrimeExplorer />;
}
