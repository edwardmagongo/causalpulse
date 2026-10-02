import { redirect } from "next/navigation";

import { getSession } from "@/lib/session";

/**
 * Clear the cookie and go to sign-in.
 *
 * A page that meets a 401 redirects here, because a render cannot clear a
 * cookie and a route handler can. Never link to this with `<Link>`: a
 * prefetch would sign the user out. The header's own control is a form that
 * calls the `signOut` action.
 */
export async function GET() {
  (await getSession()).destroy();
  redirect("/sign-in");
}
