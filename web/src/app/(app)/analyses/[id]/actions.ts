"use server";

import { requestInterpretation } from "@/lib/api/client";
import type { InterpretationChoice } from "@/lib/api/types";
import { refusal } from "@/lib/refusal";

export type InterpretResult = { ok: true } | { error: string };

/**
 * Ask for the analysis's interpretation, from the chosen entry of the
 * server's list or, with none, its default; or answer with the API's refusal.
 */
export async function askForInterpretation(
  analysisId: string,
  choice?: InterpretationChoice,
): Promise<InterpretResult> {
  try {
    await requestInterpretation(analysisId, choice);
  } catch (error) {
    return refusal(error);
  }
  return { ok: true };
}
