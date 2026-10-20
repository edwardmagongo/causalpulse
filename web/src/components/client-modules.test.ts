import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";

import { describe, expect, it } from "vitest";

const SOURCE = join(__dirname, "..");

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name);
    if (statSync(path).isDirectory()) return sourceFiles(path);
    return /\.tsx?$/.test(name) && !/\.test\.tsx?$/.test(name) ? [path] : [];
  });
}

// Next.js turns a "use client" module's exports into references a server
// component may render but not call, and vitest does not apply that split, so
// a helper exported from one passes every component test and fails the page
// with a 500. A client module therefore exports components and types only.
describe('"use client" modules', () => {
  const clientModules = sourceFiles(SOURCE).filter((path) =>
    /^\s*["']use client["']/.test(readFileSync(path, "utf8")),
  );

  it("are found", () => {
    expect(clientModules.length).toBeGreaterThan(0);
  });

  it("export nothing a server component could call", () => {
    const offending = clientModules.flatMap((path) =>
      [...readFileSync(path, "utf8").matchAll(/^export\s+(?:async\s+)?(?:function|const|let|var|class)\s+(\w+)/gm)]
        .map((match) => match[1])
        .filter((name) => !/^[A-Z]/.test(name))
        .map((name) => `${relative(SOURCE, path)}: ${name}`),
    );
    expect(offending).toEqual([]);
  });
});
