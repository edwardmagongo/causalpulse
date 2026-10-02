import { relay } from "@/lib/api/relay";

/** The replayable manifest, as a download. */
export async function GET(_request: Request, { params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return relay(`/v1/analyses/${encodeURIComponent(id)}/manifest`, {}, {
    "content-disposition": 'attachment; filename="manifest.json"',
  });
}
