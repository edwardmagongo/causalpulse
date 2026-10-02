/**
 * Leave for `/sign-out` with a full page load.
 *
 * `/sign-out` is a route handler that clears the cookie, not a page. The
 * client router would fetch it as page data, so the browser has to make the
 * request itself.
 */
export function leaveForSignOut(): void {
  // eslint-disable-next-line @next/next/no-location-assign-relative-destination -- a route handler, not a page
  window.location.href = "/sign-out";
}
