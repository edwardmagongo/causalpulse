/**
 * The schema form's state, and the JSON `POST /v1/analyses` takes as `schema`.
 *
 * The API is the only validator. This module checks that the required fields
 * are filled and nothing else; every rule about what a schema may say is
 * `schema_from_dict`'s, and its 422 is shown to the user verbatim.
 */

export type OutcomeType = "binary" | "continuous";
export type Provenance = "randomised" | "natural_experiment" | "observational";

export const OUTCOME_TYPES: OutcomeType[] = ["binary", "continuous"];
export const PROVENANCES: Provenance[] = ["randomised", "natural_experiment", "observational"];

export type SchemaForm = {
  unit: string;
  treatment: string;
  outcome: string;
  covariates: string[];
  outcomeType: OutcomeType | "";
  provenance: Provenance;
  time: string;
  cohort: string;
  negativeControl: string;
  auxiliary: string[];
};

export const EMPTY_FORM: SchemaForm = {
  unit: "",
  treatment: "",
  outcome: "",
  covariates: [],
  outcomeType: "",
  // The contract's own default: an undeclared provenance is the conservative one.
  provenance: "observational",
  time: "",
  cohort: "",
  negativeControl: "",
  auxiliary: [],
};

export type SchemaJson = {
  unit: string;
  treatment: string;
  outcome: string;
  covariates: string[];
  outcome_type: string;
  assignment_provenance: Provenance;
  time?: string;
  cohort?: string;
  negative_control_outcome?: string;
  auxiliary_columns?: string[];
};

const REQUIRED: [keyof SchemaForm, string][] = [
  ["unit", "unit"],
  ["treatment", "treatment"],
  ["outcome", "outcome"],
  ["outcomeType", "outcome type"],
];

/** The labels of the required fields still empty, in form order. */
export function missingRequired(form: SchemaForm): string[] {
  return REQUIRED.filter(([field]) => form[field] === "").map(([, label]) => label);
}

/**
 * The form as the API's schema JSON.
 *
 * `covariates` is always present, as `[]` when none is chosen: the API
 * requires the key, because a schema that omitted it would silently mean "no
 * confounders". The optional fields are left out when empty.
 */
export function toSchema(form: SchemaForm): SchemaJson {
  const schema: SchemaJson = {
    unit: form.unit,
    treatment: form.treatment,
    outcome: form.outcome,
    covariates: [...form.covariates],
    outcome_type: form.outcomeType,
    assignment_provenance: form.provenance,
  };
  if (form.time) schema.time = form.time;
  if (form.cohort) schema.cohort = form.cohort;
  if (form.negativeControl) schema.negative_control_outcome = form.negativeControl;
  if (form.auxiliary.length > 0) schema.auxiliary_columns = [...form.auxiliary];
  return schema;
}

type SingleRole = "unit" | "treatment" | "outcome";
const SINGLE_ROLES: SingleRole[] = ["unit", "treatment", "outcome"];

/**
 * The columns a role's control offers.
 *
 * A column already chosen as the unit, the treatment or the outcome is not
 * offered for another of those roles, or as a covariate. This is a
 * convenience, not a rule: the API decides what a schema may say.
 */
export function columnsFor(
  role: SingleRole | "covariates",
  form: SchemaForm,
  columns: string[],
): string[] {
  const taken = new Set(
    SINGLE_ROLES.filter((other) => other !== role)
      .map((other) => form[other])
      .filter((column) => column !== ""),
  );
  return columns.filter((column) => !taken.has(column));
}

/**
 * The form after one of unit, treatment or outcome changes: a covariate that
 * now holds that role is dropped, so the form never sends what its own
 * controls no longer offer.
 */
export function withRole(form: SchemaForm, role: SingleRole, column: string): SchemaForm {
  return {
    ...form,
    [role]: column,
    covariates: form.covariates.filter((covariate) => covariate !== column),
  };
}

/** The form with every choice that is not one of `columns` cleared, for a change of dataset. */
export function restrictedTo(form: SchemaForm, columns: string[]): SchemaForm {
  const known = new Set(columns);
  const keep = (column: string) => (known.has(column) ? column : "");
  return {
    ...form,
    unit: keep(form.unit),
    treatment: keep(form.treatment),
    outcome: keep(form.outcome),
    covariates: form.covariates.filter((column) => known.has(column)),
    time: keep(form.time),
    cohort: keep(form.cohort),
    negativeControl: keep(form.negativeControl),
    auxiliary: form.auxiliary.filter((column) => known.has(column)),
  };
}

export const MAX_SEED = 2 ** 31 - 1;

/** The seed the text field holds, or `null` when it is not an integer the API's column can store. */
export function parseSeed(text: string): number | null {
  if (!/^\d+$/.test(text.trim())) return null;
  const seed = Number(text.trim());
  return seed <= MAX_SEED ? seed : null;
}
