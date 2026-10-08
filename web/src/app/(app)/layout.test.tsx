import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const session = vi.hoisted(() => ({ current: {} as { key?: string; name?: string } }));

vi.mock("server-only", () => ({}));
// The cookie's contents; the layout calls no API.
vi.mock("@/lib/session", () => ({ getSession: async () => session.current }));
vi.mock("../sign-in/actions", () => ({ signOut: vi.fn() }));

import AppLayout from "./layout";

afterEach(cleanup);

async function renderShell(current: { key?: string; name?: string }) {
  session.current = current;
  render(await AppLayout({ children: <p>the page</p> }));
  const header = screen.getByRole("banner");
  return { header, nav: within(header).getByRole("navigation", { name: "Main" }) };
}

/** Every link and button of the header, by accessible name, in order. */
function controls(header: HTMLElement): string[] {
  return Array.from(header.querySelectorAll("a, button")).map(
    (control) => control.getAttribute("aria-label") ?? control.textContent ?? "",
  );
}

describe("the shell's header", () => {
  it("shows a signed-in user the three links, the name and Sign out", async () => {
    const { header, nav } = await renderShell({ key: "cpk_test", name: "Ada" });
    expect(within(nav).getByRole("link", { name: "Analyses" }).getAttribute("href")).toBe("/");
    expect(within(nav).getByRole("link", { name: "Benchmark study" }).getAttribute("href")).toBe("/study");
    expect(within(nav).getByRole("link", { name: "New analysis" }).getAttribute("href")).toBe("/analyses/new");
    expect(within(header).getByRole("button", { name: "Sign out" })).toBeTruthy();
    expect(within(header).getByText("Ada")).toBeTruthy();
    expect(controls(header)).toEqual(["Analyses", "Benchmark study", "New analysis", "Sign out"]);
    expect(within(header).queryByRole("link", { name: "Sign in" })).toBeNull();
    expect(screen.getByRole("main").textContent).toBe("the page");
  });

  it("shows a visitor the study link and a Sign in link, and nothing else", async () => {
    const { header, nav } = await renderShell({});
    expect(within(nav).getByRole("link", { name: "Benchmark study" }).getAttribute("href")).toBe("/study");
    expect(within(header).getByRole("link", { name: "Sign in" }).getAttribute("href")).toBe("/sign-in");
    expect(controls(header)).toEqual(["Benchmark study", "Sign in"]);
    for (const name of ["Analyses", "New analysis"]) {
      expect(within(header).queryByRole("link", { name }), name).toBeNull();
    }
    expect(within(header).queryByRole("button")).toBeNull();
    expect(header.querySelector("form")).toBeNull();
    expect(header.textContent).not.toContain("Sign out");
    expect(screen.getByRole("main").textContent).toBe("the page");
  });

  it("goes by the key, not the name: a session that holds only a name is a visitor's", async () => {
    const { header } = await renderShell({ name: "Ada" });
    expect(controls(header)).toEqual(["Benchmark study", "Sign in"]);
    expect(header.textContent).not.toContain("Ada");
  });

  it("gives the study link the same name and the same narrow label in both forms", async () => {
    const signedIn = await renderShell({ key: "cpk_test", name: "Ada" });
    const markup = within(signedIn.nav).getByRole("link", { name: "Benchmark study" }).outerHTML;
    cleanup();
    const visitor = await renderShell({});
    expect(within(visitor.nav).getByRole("link", { name: "Benchmark study" }).outerHTML).toBe(markup);
  });
});
