import { describe, expect, it } from "vitest";

import {
  EMPTY_FORM,
  columnsFor,
  missingRequired,
  parseSeed,
  restrictedTo,
  toSchema,
  withRole,
  type SchemaForm,
} from "./schema-form";

const LALONDE: SchemaForm = {
  ...EMPTY_FORM,
  unit: "unit_id",
  treatment: "treat",
  outcome: "re78",
  covariates: ["age", "educ"],
  outcomeType: "continuous",
};
const COLUMNS = ["unit_id", "treat", "re78", "age", "educ", "year"];

describe("toSchema", () => {
  it("sends covariates as an empty list when none is chosen, never omits it", () => {
    const schema = toSchema({ ...LALONDE, covariates: [] });
    expect(schema.covariates).toEqual([]);
    expect(Object.keys(schema)).toContain("covariates");
  });

  it("leaves every empty optional field out", () => {
    expect(toSchema(LALONDE)).toEqual({
      unit: "unit_id",
      treatment: "treat",
      outcome: "re78",
      covariates: ["age", "educ"],
      outcome_type: "continuous",
      assignment_provenance: "observational",
    });
  });

  it("uses the API's field names for the optional fields that are set", () => {
    const schema = toSchema({
      ...LALONDE,
      provenance: "randomised",
      time: "year",
      cohort: "g",
      negativeControl: "re74",
      auxiliary: ["proxy"],
    });
    expect(schema.assignment_provenance).toBe("randomised");
    expect(schema.time).toBe("year");
    expect(schema.cohort).toBe("g");
    expect(schema.negative_control_outcome).toBe("re74");
    expect(schema.auxiliary_columns).toEqual(["proxy"]);
  });

  it("never sends treatment_start, which no analysis reads", () => {
    expect(Object.keys(toSchema({ ...LALONDE, time: "year" }))).not.toContain("treatment_start");
  });

  it("copies the lists, so the request cannot alias the form's state", () => {
    const schema = toSchema(LALONDE);
    schema.covariates.push("extra");
    expect(LALONDE.covariates).toEqual(["age", "educ"]);
  });
});

describe("missingRequired", () => {
  it("names each empty required field in form order", () => {
    expect(missingRequired(EMPTY_FORM)).toEqual(["unit", "treatment", "outcome", "outcome type"]);
    expect(missingRequired({ ...LALONDE, outcomeType: "" })).toEqual(["outcome type"]);
  });

  it("asks for nothing else: no covariates and no optional field is fine", () => {
    expect(missingRequired({ ...LALONDE, covariates: [] })).toEqual([]);
  });
});

describe("columnsFor", () => {
  it("does not offer a column another single role already holds", () => {
    expect(columnsFor("outcome", LALONDE, COLUMNS)).toEqual(["re78", "age", "educ", "year"]);
    expect(columnsFor("covariates", LALONDE, COLUMNS)).toEqual(["age", "educ", "year"]);
  });

  it("still offers a role its own current column", () => {
    expect(columnsFor("unit", LALONDE, COLUMNS)).toContain("unit_id");
  });

  it("offers everything while nothing is chosen", () => {
    expect(columnsFor("covariates", EMPTY_FORM, COLUMNS)).toEqual(COLUMNS);
  });
});

describe("withRole", () => {
  it("drops a covariate that becomes the outcome", () => {
    const next = withRole(LALONDE, "outcome", "age");
    expect(next.outcome).toBe("age");
    expect(next.covariates).toEqual(["educ"]);
  });
});

describe("restrictedTo", () => {
  it("clears every choice the new dataset does not have and keeps the rest", () => {
    const next = restrictedTo({ ...LALONDE, time: "year", auxiliary: ["proxy", "age"] }, [
      "unit_id",
      "age",
      "year",
    ]);
    expect(next).toMatchObject({
      unit: "unit_id",
      treatment: "",
      outcome: "",
      covariates: ["age"],
      time: "year",
      auxiliary: ["age"],
      outcomeType: "continuous",
    });
  });
});

describe("parseSeed", () => {
  it("accepts integers the API's 32-bit column can hold", () => {
    expect(parseSeed("0")).toBe(0);
    expect(parseSeed(" 42 ")).toBe(42);
    expect(parseSeed("2147483647")).toBe(2147483647);
  });

  it("refuses anything else", () => {
    for (const text of ["", "-1", "1.5", "1e3", "abc", "2147483648"]) {
      expect(parseSeed(text)).toBeNull();
    }
  });
});
