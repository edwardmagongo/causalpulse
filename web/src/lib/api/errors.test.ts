import { describe, expect, it } from "vitest";

import { ApiError, ApiUnavailable, Unauthorized, detailOf, errorFor } from "./errors";

describe("detailOf", () => {
  it("passes the API's own message through verbatim", () => {
    expect(detailOf({ detail: "invalid schema: outcome_type must be one of" }, 422)).toBe(
      "invalid schema: outcome_type must be one of",
    );
  });

  it("joins FastAPI's validation items, naming each field without the body prefix", () => {
    const body = {
      detail: [
        { type: "missing", loc: ["body", "dataset_id"], msg: "Field required" },
        { type: "less_than", loc: ["body", "random_seed"], msg: "Input should be less than 2147483648" },
      ],
    };
    expect(detailOf(body, 422)).toBe(
      "dataset_id: Field required; random_seed: Input should be less than 2147483648",
    );
  });

  it("falls back to the status when the body says nothing usable", () => {
    expect(detailOf(null, 500)).toBe("the API answered 500");
    expect(detailOf({ detail: [] }, 422)).toBe("the API answered 422");
    expect(detailOf("<html>", 502)).toBe("the API answered 502");
  });

  it("says a 429 with nothing usable in its body is too many requests", () => {
    expect(detailOf(null, 429)).toBe("Too many requests. Try again later.");
    expect(detailOf("<html>", 429)).toBe("Too many requests. Try again later.");
    expect(detailOf({ detail: "Too many uploads today for this account. Try again in 3 hours." }, 429)).toBe(
      "Too many uploads today for this account. Try again in 3 hours.",
    );
  });
});

describe("errorFor", () => {
  it("maps 401 to Unauthorized, carrying the API's own detail", () => {
    const error = errorFor(401, { detail: "x" });
    expect(error).toBeInstanceOf(Unauthorized);
    expect((error as Unauthorized).detail).toBe("x");
  });

  it("gives an Unauthorized with no body a default detail", () => {
    expect(new Unauthorized().detail).toBe("unauthorized");
    expect((errorFor(401, null) as Unauthorized).detail).toBe("the API answered 401");
  });

  it("maps 503 to ApiUnavailable whatever the body says", () => {
    expect(errorFor(503, { detail: "service unavailable: OperationalError" })).toBeInstanceOf(
      ApiUnavailable,
    );
  });

  it("keeps the status and the detail of every other refusal", () => {
    const error = errorFor(413, { detail: "the upload exceeds the 52428800-byte limit" });
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).status).toBe(413);
    expect((error as ApiError).detail).toBe("the upload exceeds the 52428800-byte limit");
  });

  it("does not treat a 404 or a 500 as unavailable", () => {
    expect(errorFor(404, { detail: "no analysis x" })).toBeInstanceOf(ApiError);
    expect(errorFor(500, null)).toBeInstanceOf(ApiError);
  });
});
