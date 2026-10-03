import { describe, expect, it, vi } from "vitest";

const relay = vi.hoisted(() => vi.fn(async () => new Response("relayed", { status: 201 })));
vi.mock("@/lib/api/relay", () => ({ relay }));

import { POST } from "./route";

function post(headers: Record<string, string>) {
  return POST(new Request("http://localhost:3000/api/datasets", { method: "POST", headers, body: "x" }));
}

describe("the upload relay's Origin check", () => {
  it("relays a post whose Origin is this site", async () => {
    const response = await post({ origin: "http://localhost:3000", host: "localhost:3000" });
    expect(response.status).toBe(201);
  });

  it("refuses a foreign Origin", async () => {
    const response = await post({ origin: "https://evil.example", host: "localhost:3000" });
    expect(response.status).toBe(403);
  });

  it("refuses the literal null a sandboxed cross-site form sends, with the 403 JSON", async () => {
    const response = await post({ origin: "null", host: "localhost:3000" });
    expect(response.status).toBe(403);
    expect(await response.json()).toEqual({ detail: "the upload must come from this site" });
  });

  it("refuses a post with no Origin", async () => {
    expect((await post({ host: "localhost:3000" })).status).toBe(403);
  });

  it("calls the API only for the post it accepted", async () => {
    relay.mockClear();
    await post({ origin: "null", host: "localhost:3000" });
    expect(relay).not.toHaveBeenCalled();
  });
});
