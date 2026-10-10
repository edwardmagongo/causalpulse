"use server";

import { redirect } from "next/navigation";

import { createAnalysis, getDataset } from "@/lib/api/client";
import { refusal } from "@/lib/refusal";
import type { SchemaJson } from "@/lib/schema-form";

export type ColumnsResult = { columns: string[] } | { error: string };
export type SubmitResult = { error: string };

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
