import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { datasetColumns } from "@/app/(app)/analyses/new/actions";
import type { DatasetItem, EstimatorEntry } from "@/lib/api/types";
import { leaveForSignOut } from "@/lib/sign-out";

import { NewAnalysisForm } from "./NewAnalysisForm";

vi.mock("@/app/(app)/analyses/new/actions", () => ({
  datasetColumns: vi.fn(),
  submitAnalysis: vi.fn(),
}));

vi.mock("@/lib/sign-out", () => ({ leaveForSignOut: vi.fn() }));

const columnsOf = vi.mocked(datasetColumns);

const A: DatasetItem = {
  id: "id-a",
  filename: "a.csv",
  sha256: "a".repeat(64),
  n_bytes: 1024,
  created_at: "2026-10-03T10:00:00Z",
};
const B: DatasetItem = { ...A, id: "id-b", filename: "b.csv", sha256: "b".repeat(64) };
const ESTIMATORS: EstimatorEntry[] = [{ name: "lin_regression", family: "cross_sectional" }];

function renderForm() {
  render(<NewAnalysisForm datasets={[A, B]} estimators={ESTIMATORS} />);
  return {
    existing: screen.getByLabelText("or use an earlier upload") as HTMLSelectElement,
    file: screen.getByLabelText("Upload a CSV") as HTMLInputElement,
  };
}

function pick(select: HTMLSelectElement, id: string) {
  fireEvent.change(select, { target: { value: id } });
}

function stubUpload(body: { id: string; filename: string; columns: string[] }) {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(JSON.stringify({ ...body, n_bytes: 10, sha256: "c".repeat(64) }), {
          status: 201,
          headers: { "content-type": "application/json" },
        }),
    ),
  );
}

/** jsdom does not keep a chosen file in `value` as a browser does; do it by hand. */
function emulateChosenFile(input: HTMLInputElement, name: string) {
  let chosen = `C:\\fakepath\\${name}`;
  Object.defineProperty(input, "value", {
    configurable: true,
    get: () => chosen,
    set: (next: string) => {
      chosen = next;
    },
  });
}

function jsonResponse(body: unknown, status: number) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });
}

beforeEach(() => {
  columnsOf.mockReset();
  vi.mocked(leaveForSignOut).mockReset();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("the earlier-upload select", () => {
  it("shows the chosen dataset's columns and holds its id", async () => {
    columnsOf.mockResolvedValue({ columns: ["x", "y", "z"] });
    const { existing } = renderForm();
    pick(existing, A.id);
    expect(await screen.findByText("a.csv: 3 columns")).toBeTruthy();
    expect(existing.value).toBe(A.id);
  });

  it("returns to the placeholder after an upload, and a repeat pick of the same dataset works", async () => {
    columnsOf.mockResolvedValue({ columns: ["x", "y", "z"] });
    const { existing, file } = renderForm();
    pick(existing, A.id);
    await screen.findByText("a.csv: 3 columns");

    stubUpload({ id: "id-new", filename: "new.csv", columns: ["p", "q"] });
    fireEvent.change(file, { target: { files: [new File(["p,q\n1,2\n"], "new.csv")] } });
    expect(await screen.findByText("new.csv: 2 columns")).toBeTruthy();
    expect(existing.value).toBe("");

    pick(existing, A.id);
    expect(await screen.findByText("a.csv: 3 columns")).toBeTruthy();
    expect(columnsOf).toHaveBeenCalledTimes(2);
    expect(existing.value).toBe(A.id);
  });

  it("shows a failed pick's refusal, returns to the placeholder, and lets the same pick be retried", async () => {
    columnsOf.mockResolvedValueOnce({ error: "the dataset could not be read" });
    const { existing } = renderForm();
    pick(existing, A.id);
    const alert = await screen.findByText("the dataset could not be read");
    expect(alert.getAttribute("role")).toBe("alert");
    expect(existing.value).toBe("");

    columnsOf.mockResolvedValueOnce({ columns: ["x", "y", "z"] });
    pick(existing, A.id);
    expect(await screen.findByText("a.csv: 3 columns")).toBeTruthy();
    expect(columnsOf).toHaveBeenCalledTimes(2);
    expect(screen.queryByText("the dataset could not be read")).toBeNull();
  });

  it("drops a late response for an earlier pick", async () => {
    let resolveA: (value: { columns: string[] }) => void = () => {};
    columnsOf.mockImplementation((id) =>
      id === A.id
        ? new Promise((resolve) => {
            resolveA = resolve;
          })
        : Promise.resolve({ columns: ["u", "v"] }),
    );
    const { existing } = renderForm();
    pick(existing, A.id);
    pick(existing, B.id);
    expect(await screen.findByText("b.csv: 2 columns")).toBeTruthy();

    resolveA({ columns: ["x", "y", "z"] });
    // Let the late response be handled before asserting it changed nothing.
    await waitFor(() => expect(columnsOf).toHaveBeenCalledTimes(2));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.getByText("b.csv: 2 columns")).toBeTruthy();
    expect(screen.queryByText("a.csv: 3 columns")).toBeNull();
    expect(existing.value).toBe(B.id);
  });

  it("shows an alert when the call itself is rejected", async () => {
    columnsOf.mockRejectedValueOnce(new Error("network down"));
    const { existing } = renderForm();
    pick(existing, A.id);
    const alert = await screen.findByText(
      "The dataset could not be loaded. Check the connection and try again.",
    );
    expect(alert.getAttribute("role")).toBe("alert");
    expect(existing.value).toBe("");
  });

  it("clears an earlier alert as soon as the next pick starts, not when it finishes", async () => {
    columnsOf.mockRejectedValueOnce(new Error("network down"));
    const { existing } = renderForm();
    pick(existing, A.id);
    await screen.findByText("The dataset could not be loaded. Check the connection and try again.");

    let resolveB: (value: { columns: string[] }) => void = () => {};
    columnsOf.mockImplementationOnce(
      () =>
        new Promise((resolve) => {
          resolveB = resolve;
        }),
    );
    pick(existing, B.id);
    // The second call is still pending, so only the start of the pick can have cleared it.
    await waitFor(() => expect(columnsOf).toHaveBeenCalledTimes(2));
    expect(screen.queryByText(/could not be loaded/)).toBeNull();

    resolveB({ columns: ["x", "y"] });
    expect(await screen.findByText("b.csv: 2 columns")).toBeTruthy();
  });

  it("ignores a rejection that arrives after a newer pick has succeeded", async () => {
    let rejectA: (reason: Error) => void = () => {};
    columnsOf.mockImplementation((id) =>
      id === A.id
        ? new Promise((_resolve, reject) => {
            rejectA = reject;
          })
        : Promise.resolve({ columns: ["u", "v"] }),
    );
    const { existing } = renderForm();
    pick(existing, A.id);
    pick(existing, B.id);
    expect(await screen.findByText("b.csv: 2 columns")).toBeTruthy();

    rejectA(new Error("network down"));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(screen.queryByText(/could not be loaded/)).toBeNull();
    expect(screen.getByText("b.csv: 2 columns")).toBeTruthy();
    expect(existing.value).toBe(B.id);
  });

  it("ignores an upload whose request fails after a later pick was made", async () => {
    let failUpload: (reason: Error) => void = () => {};
    vi.stubGlobal(
      "fetch",
      vi.fn(
        () =>
          new Promise<Response>((_resolve, reject) => {
            failUpload = reject;
          }),
      ),
    );
    columnsOf.mockResolvedValue({ columns: ["u", "v"] });
    const { existing, file } = renderForm();
    fireEvent.change(file, { target: { files: [new File(["p,q\n1,2\n"], "new.csv")] } });
    await screen.findByText("Uploading.");
    pick(existing, B.id);
    expect(await screen.findByText("b.csv: 2 columns")).toBeTruthy();

    failUpload(new Error("connection reset"));
    await waitFor(() => expect(screen.queryByText("Uploading.")).toBeNull());
    expect(screen.queryByText(/could not be sent/)).toBeNull();
    expect(screen.getByText("b.csv: 2 columns")).toBeTruthy();
    expect(existing.value).toBe(B.id);
  });

  it("keeps a later pick when an earlier upload finishes after it", async () => {
    let finishUpload: (response: Response) => void = () => {};
    vi.stubGlobal(
      "fetch",
      vi.fn(
        () =>
          new Promise<Response>((resolve) => {
            finishUpload = resolve;
          }),
      ),
    );
    columnsOf.mockResolvedValue({ columns: ["u", "v"] });
    const { existing, file } = renderForm();
    fireEvent.change(file, { target: { files: [new File(["p,q\n1,2\n"], "new.csv")] } });
    expect(await screen.findByText("Uploading.")).toBeTruthy();

    pick(existing, B.id);
    expect(await screen.findByText("b.csv: 2 columns")).toBeTruthy();

    finishUpload(
      new Response(
        JSON.stringify({ id: "id-new", filename: "new.csv", columns: ["p", "q", "r"], n_bytes: 10, sha256: "c".repeat(64) }),
        { status: 201, headers: { "content-type": "application/json" } },
      ),
    );
    // The upload's end is visible as the status line going away.
    await waitFor(() => expect(screen.queryByText("Uploading.")).toBeNull());
    expect(screen.getByText("b.csv: 2 columns")).toBeTruthy();
    expect(screen.queryByText("new.csv: 3 columns")).toBeNull();
    expect(existing.value).toBe(B.id);
  });
});

describe("the file input", () => {
  it("is emptied after a failed upload, so the same file can be chosen again", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ detail: "the upload exceeds the limit" }, 413)));
    const { file } = renderForm();
    emulateChosenFile(file, "big.csv");
    fireEvent.change(file, { target: { files: [new File(["p,q\n1,2\n"], "big.csv")] } });
    await screen.findByText("the upload exceeds the limit");
    expect(file.value).toBe("");
  });

  it("is emptied after a successful upload too", async () => {
    stubUpload({ id: "id-new", filename: "new.csv", columns: ["p", "q"] });
    const { file } = renderForm();
    emulateChosenFile(file, "new.csv");
    fireEvent.change(file, { target: { files: [new File(["p,q\n1,2\n"], "new.csv")] } });
    await screen.findByText("new.csv: 2 columns");
    expect(file.value).toBe("");
  });

  it("is emptied when the upload is answered 401, which leaves for sign-out", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ detail: "signed out" }, 401)));
    const { file } = renderForm();
    emulateChosenFile(file, "new.csv");
    fireEvent.change(file, { target: { files: [new File(["p,q\n1,2\n"], "new.csv")] } });
    await waitFor(() => expect(leaveForSignOut).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(screen.queryByText("Uploading.")).toBeNull());
    expect(file.value).toBe("");
  });
});
