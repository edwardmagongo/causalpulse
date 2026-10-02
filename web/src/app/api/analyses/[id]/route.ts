import { relay } from "@/lib/api/relay";

/** The analysis as the API returns it, for the page's polling. */
export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return relay(`/v1/analyses/${encodeURIComponent(id)}`);
}
