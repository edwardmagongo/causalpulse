/** What a call to cp-api can fail with. Kept free of server-only imports so it can be tested. */

/**
 * 401: the key is missing, wrong or revoked.
 *
 * `detail` is the API's own words when the response had a body; a call that
 * has none (no session, or a relayed response) uses the default.
 */
export class Unauthorized extends Error {
  constructor(readonly detail: string = "unauthorized") {
    super(detail);
    this.name = "Unauthorized";
  }
}

/** 503, or no connection at all. */
export class ApiUnavailable extends Error {
  constructor() {
    super("the service is unavailable");
    this.name = "ApiUnavailable";
  }
}

/** Any other refusal, with the API's own `detail`. */
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly detail: string,
  ) {
    super(detail);
    this.name = "ApiError";
  }
}

/**
 * The `detail` of an error body as one line of text.
 *
 * The API's own refusals carry a string. FastAPI's request validation carries
 * a list of `{loc, msg}` items instead, one per offending field.
 */
export function detailOf(body: unknown, status: number): string {
  if (typeof body === "object" && body !== null && "detail" in body) {
    const detail = (body as { detail: unknown }).detail;
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const lines = detail.map((item) => {
        const { loc, msg } = item as { loc?: unknown[]; msg?: unknown };
        const field = Array.isArray(loc) ? loc.filter((part) => part !== "body").join(".") : "";
        return field ? `${field}: ${String(msg)}` : String(msg);
      });
      if (lines.length > 0) return lines.join("; ");
    }
  }
  return `the API answered ${status}`;
}

/** The error a non-2xx status stands for. */
export function errorFor(status: number, body: unknown): Error {
  if (status === 401) return new Unauthorized(detailOf(body, 401));
  if (status === 503) return new ApiUnavailable();
  return new ApiError(status, detailOf(body, status));
}
