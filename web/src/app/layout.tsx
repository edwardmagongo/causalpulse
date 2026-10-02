import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "CausalPulse",
  description: "Upload a dataset, declare its schema, and read the estimates with their evidence.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
