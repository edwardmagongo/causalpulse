import { connection } from "next/server";

// Rendered per request, so its scripts carry the request's nonce.
export default async function SignInLayout({ children }: { children: React.ReactNode }) {
  await connection();
  return children;
}
