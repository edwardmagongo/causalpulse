import { beforeEach, describe, expect, it, vi } from "vitest";

import { ApiError, ApiUnavailable, Unauthorized } from "./errors";

vi.mock("server-only", () => ({}));
vi.mock("next/navigation", () => ({
  notFound: () => {
    throw new Error("NOT_FOUND");
  },
  redirect: (to: string) => {
    throw new Error(`REDIRECT ${to}`);
  },
}));

import { load } from "./load";

function failing(error: unknown) {
  return () => Promise.reject(error);
}

beforeEach(() => vi.clearAllMocks());

describe("load", () => {
  it("returns the data of a successful read", async () => {
    expect(await load(async () => 7)).toEqual({ ok: true, data: 7 });
  });

  it("sends a 401 to /sign-out", async () => {
    await expect(load(failing(new Unauthorized("x")))).rejects.toThrow("REDIRECT /sign-out");
  });

  it("shows not-found for a 404", async () => {
    await expect(load(failing(new ApiError(404, "no analysis x")))).rejects.toThrow("NOT_FOUND");
  });

  it("shows not-found for a 422, which on a read means a malformed id or cursor", async () => {
    await expect(load(failing(new ApiError(422, "id: Input should be a valid UUID")))).rejects.toThrow(
      "NOT_FOUND",
    );
  });

  it("answers an unavailable API as a value, so the page can say so in place", async () => {
    expect(await load(failing(new ApiUnavailable()))).toEqual({ ok: false, reason: "unavailable" });
  });

  it("rethrows anything else", async () => {
    const other = new ApiError(500, "boom");
    await expect(load(failing(other))).rejects.toBe(other);
    const plain = new Error("bug");
    await expect(load(failing(plain))).rejects.toBe(plain);
  });
});
