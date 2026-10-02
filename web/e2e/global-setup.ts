/**
 * Wait until the stack answers, so no test spends its own timeout on a start-up.
 *
 * `docker compose up -d` returns once the containers have started, which is
 * before the API and the web server accept requests.
 */
const WEB = process.env.CP_WEB_URL ?? "http://localhost:3000";
const API_HEALTH = process.env.CP_E2E_API_HEALTH ?? "";
const DEADLINE_MS = 120_000;

async function answers(url: string): Promise<boolean> {
  try {
    return (await fetch(url, { redirect: "manual" })).ok;
  } catch {
    return false;
  }
}

export default async function globalSetup(): Promise<void> {
  const urls = [`${WEB}/sign-in`, ...(API_HEALTH ? [API_HEALTH] : [])];
  const started = Date.now();
  for (const url of urls) {
    while (!(await answers(url))) {
      if (Date.now() - started > DEADLINE_MS) {
        throw new Error(`${url} did not answer within ${DEADLINE_MS / 1000} s`);
      }
      await new Promise((done) => setTimeout(done, 1000));
    }
  }
}
