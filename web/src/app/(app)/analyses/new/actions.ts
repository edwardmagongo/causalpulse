"use server";

import { redirect } from "next/navigation";

import { createAnalysis, getDataset } from "@/lib/api/client";
import { ApiError, ApiUnavailable, Unauthorized } from "@/lib/api/errors";
import type { SchemaJson } from "@/lib/schema-form";

export type ColumnsResult = { columns: string[] } | { error: string };
export type SubmitResult = { error: string };

const UNAVAILABLE = "The service is unavailable. Try again in a moment.";

/** What an action answers for a refusal; a 401 leaves for sign-out instead. */
function refusal(error: unknown): { error: string } {
  if (error instanceof Unauthorized) redirect("/sign-out");
  if (error instanceof ApiUnavailable) return { error: UNAVAILABLE };
  if (error instanceof ApiError) return { error: error.detail };
  throw error;
}

/** The columns of an earlier upload, for the schema form. */
export async function datasetColumns(datasetId: string): Promise<ColumnsResult> {
  try {
    return { columns: (await getDataset(datasetId)).columns };
  } catch (error) {
    return refusal(error);
  }
}

/** Queue the analysis and go to its page, or answer with the API's refusal. */
export async function submitAnalysis(request: {
  datasetId: string;
  schema: SchemaJson;
  estimators: string[] | null;
  seed: number;
}): Promise<SubmitResult> {
  let id: string;
  try {
    const created = await createAnalysis({
      dataset_id: request.datasetId,
      schema: request.schema,
      estimators: request.estimators,
      random_seed: request.seed,
    });
    id = created.id;
  } catch (error) {
    return refusal(error);
  }
  redirect(`/analyses/${id}`);
}
