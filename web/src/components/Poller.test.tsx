import { act, cleanup, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { leaveForSignOut } from "@/lib/sign-out";

import { Poller } from "./Poller";

// One router object for every render, as Next.js gives: the poller's effect
// depends on it, so a fresh object per render would restart the timers.
const router = vi.hoisted(() => ({ refresh: vi.fn() }));

vi.mock("next/navigation", () => ({ useRouter: () => router }));
vi.mock("@/lib/sign-out", () => ({ leaveForSignOut: vi.fn() }));

const UNAVAILABLE = "The service is unavailable. Still checking.";
const IN_PROGRESS = "In progress. This page updates when the analysis finishes.";

function respond(status: number, body: unknown = {}) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

/** Let the clock run `ms` and the promises it releases settle. */
async function elapse(ms: number) {
  await act(async () => {
    await vi.advanceTimersByTimeAsync(ms);
  });
}

beforeEach(() => {
  vi.useFakeTimers();
  router.refresh.mockReset();
  vi.mocked(leaveForSignOut).mockReset();
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

describe("Poller", () => {
  it("backs off between polls, says when the service is down, and refreshes once on completion", async () => {
    const fetchMock = vi
      .fn()
      .mockResolvedValueOnce(respond(503))
      .mockResolvedValueOnce(respond(200, { status: "running" }))
      .mockResolvedValueOnce(respond(200, { status: "succeeded" }));
    vi.stubGlobal("fetch", fetchMock);
    render(<Poller analysisId="a1" />);
    expect(screen.getByRole("status").textContent).toBe(IN_PROGRESS);

    // The first poll is 2000 ms after mounting.
    await elapse(1999);
    expect(fetchMock).toHaveBeenCalledTimes(0);
    await elapse(1);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe("/api/analyses/a1");
    expect(screen.getByRole("status").textContent).toBe(UNAVAILABLE);

    // The second, 3000 ms after the first.
    await elapse(2999);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    await elapse(1);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(screen.getByRole("status").textContent).toBe(IN_PROGRESS);
    expect(router.refresh).not.toHaveBeenCalled();

    // The third, 4500 ms after the second, finds it finished.
    await elapse(4499);
    expect(fetchMock).toHaveBeenCalledTimes(2);
    await elapse(1);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(router.refresh).toHaveBeenCalledTimes(1);

    // Nothing more is scheduled once the page has been told to refresh.
    expect(vi.getTimerCount()).toBe(0);
    await elapse(60_000);
    expect(fetchMock).toHaveBeenCalledTimes(3);
    expect(router.refresh).toHaveBeenCalledTimes(1);
  });

  it("leaves for sign-out on a 401 and polls no more", async () => {
    const fetchMock = vi.fn().mockResolvedValue(respond(401, { detail: "missing or invalid API key" }));
    vi.stubGlobal("fetch", fetchMock);
    render(<Poller analysisId="a1" />);

    await elapse(2000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(leaveForSignOut).toHaveBeenCalledTimes(1);
    expect(vi.getTimerCount()).toBe(0);

    await elapse(60_000);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(router.refresh).not.toHaveBeenCalled();
  });

  it("does nothing when a poll resolves after the component has unmounted", async () => {
    // The response has arrived but its body has not: the poll is awaiting `json()`.
    let finish: (body: unknown) => void = () => {};
    const body = new Promise((resolve) => {
      finish = resolve;
    });
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: () => body });
    vi.stubGlobal("fetch", fetchMock);
    const { unmount } = render(<Poller analysisId="a1" />);

    await elapse(2000);
    expect(fetchMock).toHaveBeenCalledTimes(1);

    unmount();
    await act(async () => {
      finish({ status: "succeeded" });
    });

    expect(router.refresh).not.toHaveBeenCalled();
    expect(vi.getTimerCount()).toBe(0);
  });
});
