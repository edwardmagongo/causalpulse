import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import GlobalError from "./global-error";

/** The page as a string: it is a whole document, which a container element cannot hold. */
function markupOf(error: Error & { digest?: string }): string {
  return renderToStaticMarkup(<GlobalError error={error} retry={() => {}} />);
}

describe("GlobalError", () => {
  it("is its own document, with a title and the dashboard's two sentences", () => {
    const markup = markupOf(new Error("secret detail"));
    expect(markup.startsWith('<html lang="en">')).toBe(true);
    expect(markup).toContain("<title>CausalPulse</title>");
    expect(markup).toContain("Something went wrong on this page.");
    expect(markup).toContain("Nothing was changed by this. You can try again.");
    expect(markup).not.toContain("Reference");
    expect(markup).not.toContain("secret detail");
  });

  it("shows the digest and never the error's own message", () => {
    const error: Error & { digest?: string } = new Error("secret detail");
    error.digest = "abc123";
    const markup = markupOf(error);
    expect(markup).toContain("Reference: abc123");
    expect(markup).not.toContain("secret detail");
  });
});
