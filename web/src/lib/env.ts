/** The two settings the server cannot run without. Read at start and at use. */
export function sessionSecret(): string {
  const secret = process.env.CP_SESSION_SECRET;
  if (!secret || secret.length < 32) {
    throw new Error("CP_SESSION_SECRET must be set to at least 32 characters");
  }
  return secret;
}

export function apiUrl(): string {
  const url = process.env.CP_API_URL;
  if (!url) throw new Error("CP_API_URL must be set to the cp-api base URL");
  return url.replace(/\/$/, "");
}
