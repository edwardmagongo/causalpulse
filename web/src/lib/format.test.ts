import { describe, expect, it } from "vitest";

import {
  formatBytes,
  formatInterval,
  formatLevel,
  formatNumber,
  formatP,
  formatSeconds,
  formatTime,
} from "./format";

describe("formatNumber", () => {
  it("renders an absent value as a dash, never as zero", () => {
    expect(formatNumber(null)).toBe("—");
    expect(formatNumber(undefined)).toBe("—");
    expect(formatNumber(0)).toBe("0");
  });

  it("keeps four significant figures", () => {
    expect(formatNumber(-8332.866670436902)).toBe("-8333");
    expect(formatNumber(0.012345678)).toBe("0.01235");
    expect(formatNumber(1.5)).toBe("1.5");
    expect(formatNumber(-19.513)).toBe("-19.51");
  });

  it("switches to exponent form at the extremes instead of printing a run of zeros", () => {
    expect(formatNumber(12345678)).toBe("1.235e+7");
    expect(formatNumber(0.00001234)).toBe("1.234e-5");
  });
});

describe("formatP", () => {
  it("bounds a tiny p-value instead of rounding it to zero", () => {
    expect(formatP(0.00004)).toBe("< 0.001");
    expect(formatP(0)).toBe("< 0.001");
    expect(formatP(0.0312)).toBe("0.031");
    expect(formatP(null)).toBe("—");
  });
});

describe("formatInterval", () => {
  it("is a dash unless both ends are present", () => {
    expect(formatInterval(-1.5, 2.25)).toBe("[-1.5, 2.25]");
    expect(formatInterval(null, 2)).toBe("—");
    expect(formatInterval(1, null)).toBe("—");
  });

  it("keeps an end that is exactly zero", () => {
    expect(formatInterval(0, 2)).toBe("[0, 2]");
  });
});

describe("the rest", () => {
  it("formats a level, a time, a duration and a size", () => {
    expect(formatLevel(0.95)).toBe("95%");
    expect(formatLevel(0.9)).toBe("90%");
    expect(formatTime("2026-10-01T12:30:15.123456+03:00")).toBe("2026-10-01 09:30:15 UTC");
    expect(formatTime(null)).toBe("—");
    expect(formatTime("not a time")).toBe("—");
    expect(formatSeconds(0.4123)).toBe("0.41 s");
    expect(formatSeconds(125)).toBe("2 min 5 s");
    expect(formatSeconds(null)).toBe("—");
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(844624)).toBe("825 KB");
    expect(formatBytes(52428793)).toBe("50 MB");
  });
});
