import { describe, expect, it } from "vitest";

import { config } from "./proxy";

// Next.js anchors a matcher to the whole path; do the same here.
const gated = new RegExp(`^${config.matcher[0]}$`);

describe("the session gate's matcher", () => {
  it("gates every page, including paths that only start like an exclusion", () => {
    for (const path of ["/", "/analyses/new", "/analyses/abc", "/study", "/api-keys", "/apixyz", "/sign-insider", "/sign-outside"]) {
      expect(gated.test(path), path).toBe(true);
    }
  });

  it("leaves out the relay handlers, the sign-in and sign-out routes, and static files", () => {
    for (const path of [
      "/api",
      "/api/datasets",
      "/api/analyses/abc/report",
      "/sign-in",
      "/sign-out",
      "/_next/static/x.js",
      "/_next/image",
      "/favicon.ico",
    ]) {
      expect(gated.test(path), path).toBe(false);
    }
  });
});
