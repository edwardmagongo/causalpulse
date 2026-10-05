/**
 * The Content-Security-Policy every page is served with.
 *
 * Scripts run only with this request's nonce (Next.js puts it on its own
 * scripts when it finds the policy on the request). `strict-dynamic` lets
 * those scripts load the chunks they need. Styles allow inline: the matrix
 * and the plot set custom properties in `style` attributes, and an evidence
 * report is shown in a sandboxed `srcdoc` frame, which inherits this policy
 * and carries its own `<style>` and `data:` images.
 */
export function contentSecurityPolicy(nonce: string, development: boolean): string {
  return [
    "default-src 'self'",
    `script-src 'self' 'nonce-${nonce}' 'strict-dynamic'${development ? " 'unsafe-eval'" : ""}`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data:",
    "font-src 'self'",
    "connect-src 'self'",
    "object-src 'none'",
    "base-uri 'self'",
    "form-action 'self'",
    "frame-ancestors 'none'",
  ].join("; ");
}
