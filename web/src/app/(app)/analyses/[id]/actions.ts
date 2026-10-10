"use server";

import { requestInterpretation } from "@/lib/api/client";
import { refusal } from "@/lib/refusal";

export type InterpretResult = { ok: true } | { error: string };

/** Ask for the analysis's interpretation, or answer with the API's refusal. */
export async function askForInterpretation(analysisId: string): Promise<InterpretResult> {
  try {
    await requestInterpretation(analysisId);
  } catch (error) {
    return refusal(error);
  }
  return { ok: true };
}
