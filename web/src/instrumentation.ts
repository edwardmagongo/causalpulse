// Runs once, before the server takes a request. The check lives in its own
// module, imported only under Node.js, because it ends the process and the
// build compiles this file for the Edge runtime as well.
export async function register() {
  if (process.env.NEXT_RUNTIME === "nodejs") {
    const { requireSettings } = await import("@/lib/startup");
    requireSettings();
  }
}
