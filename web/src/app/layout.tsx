import type { Metadata } from "next";
import { connection } from "next/server";

import "./globals.css";

export const metadata: Metadata = {
  title: "CausalPulse",
  description: "Upload a dataset, declare its schema, and read the estimates with their evidence.",
};

// The policy's nonce is per request, so every page is rendered per request, the 404 included.
export default async function RootLayout({ children }: { children: React.ReactNode }) {
  await connection();
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
