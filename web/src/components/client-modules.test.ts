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
// The directive may follow comments, as Next.js allows.
function isClientModule(text: string): boolean {
  const code = text.replace(/^(?:\s*(?:\/\/[^\n]*|\/\*[\s\S]*?\*\/))*\s*/, "");
  return /^["']use client["']/.test(code);
}

// A component (PascalCase function, default or named), a type or an interface.
const ALLOWED = /^export\s+(?:(?:default\s+)?function\s+[A-Z]\w*|type\s|interface\s)/;

describe('"use client" modules', () => {
  const clientModules = sourceFiles(SOURCE).filter((path) =>
    isClientModule(readFileSync(path, "utf8")),
  );

  it("are found, the button that offers the interpretation choices among them", () => {
    expect(clientModules.map((path) => relative(SOURCE, path))).toContain(
      "components/InterpretButton.tsx",
    );
  });

  it("export nothing a server component could call", () => {
    const offending = clientModules.flatMap((path) =>
      readFileSync(path, "utf8")
        .split("\n")
        .filter((line) => /^export\b/.test(line) && !ALLOWED.test(line))
        .map((line) => `${relative(SOURCE, path)}: ${line}`),
    );
    expect(offending).toEqual([]);
  });
});
