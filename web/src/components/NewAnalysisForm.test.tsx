import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { datasetColumns } from "@/app/(app)/analyses/new/actions";
import type { DatasetItem, EstimatorEntry } from "@/lib/api/types";

import { NewAnalysisForm } from "./NewAnalysisForm";

vi.mock("@/app/(app)/analyses/new/actions", () => ({
  datasetColumns: vi.fn(),
  submitAnalysis: vi.fn(),
}));

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

beforeEach(() => {
  columnsOf.mockReset();
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
});
