"use client";

import { useState, useTransition, type ChangeEvent, type FormEvent } from "react";

import { datasetColumns, submitAnalysis } from "@/app/(app)/analyses/new/actions";
import { detailOf } from "@/lib/api/errors";
import type { DatasetItem, DatasetUploaded, EstimatorEntry, Family } from "@/lib/api/types";
import { formatBytes } from "@/lib/format";
import { leaveForSignOut } from "@/lib/sign-out";
import {
  EMPTY_FORM,
  OUTCOME_TYPES,
  PROVENANCES,
  columnsFor,
  missingRequired,
  parseSeed,
  restrictedTo,
  toSchema,
  withRole,
  type OutcomeType,
  type Provenance,
  type SchemaForm,
} from "@/lib/schema-form";

const FAMILY_LABELS: Record<Family, string> = {
  cross_sectional: "Cross-sectional",
  panel: "Panel",
  donor_pool: "Donor pool",
};
const FAMILIES: Family[] = ["cross_sectional", "panel", "donor_pool"];

type Chosen = { id: string; filename: string; columns: string[] };

// Two columns from `sm` up; one on a phone.
const FIELD_GRID = "grid gap-x-6 gap-y-5 sm:grid-cols-2";
const FIELD_BOX = "flex flex-col gap-1.5";

export function NewAnalysisForm({
  datasets,
  estimators,
}: {
  datasets: DatasetItem[];
  estimators: EstimatorEntry[];
}) {
  const [dataset, setDataset] = useState<Chosen | null>(null);
  const [datasetError, setDatasetError] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [form, setForm] = useState<SchemaForm>(EMPTY_FORM);
  const [routerChooses, setRouterChooses] = useState(true);
  const [picked, setPicked] = useState<string[]>([]);
  const [seedText, setSeedText] = useState("0");
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [submitting, startSubmit] = useTransition();

  function choose(next: Chosen) {
    setDataset(next);
    setDatasetError(null);
    setForm((current) => restrictedTo(current, next.columns));
  }

  async function upload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setDatasetError(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch("/api/datasets", { method: "POST", body });
      if (response.status === 401) {
        leaveForSignOut();
        return;
      }
      const payload: unknown = await response.json().catch(() => null);
      if (!response.ok) {
        setDataset(null);
        setDatasetError(detailOf(payload, response.status));
        return;
      }
      const uploaded = payload as DatasetUploaded;
      choose({ id: uploaded.id, filename: uploaded.filename, columns: uploaded.columns });
    } catch {
      setDataset(null);
      setDatasetError("The upload could not be sent. Check the connection and try again.");
    } finally {
      setUploading(false);
    }
  }

  async function pickExisting(event: ChangeEvent<HTMLSelectElement>) {
    const item = datasets.find((candidate) => candidate.id === event.target.value);
    if (!item) {
      setDataset(null);
      return;
    }
    const result = await datasetColumns(item.id);
    if ("error" in result) {
      setDataset(null);
      setDatasetError(result.error);
      return;
    }
    choose({ id: item.id, filename: item.filename, columns: result.columns });
  }

  function toggle(list: string[], value: string, on: boolean): string[] {
    return on ? [...list, value] : list.filter((entry) => entry !== value);
  }

  const columns = dataset?.columns ?? [];
  const missing = missingRequired(form);
  const seed = parseSeed(seedText);
  const nonePicked = !routerChooses && picked.length === 0;
  const blocked = !dataset || missing.length > 0 || seed === null || nonePicked;

  function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (blocked || !dataset || seed === null) return;
    setSubmitError(null);
    startSubmit(async () => {
      const result = await submitAnalysis({
        datasetId: dataset.id,
        schema: toSchema(form),
        estimators: routerChooses ? null : picked,
        seed,
      });
      // On success the action redirects and never returns here.
      if (result?.error) setSubmitError(result.error);
    });
  }

  const single = (
    id: string,
    label: string,
    value: string,
    options: string[],
    onChange: (column: string) => void,
    optional = false,
  ) => (
    <div className={FIELD_BOX}>
      <label htmlFor={id} className="label">
        {label}
      </label>
      <select
        id={id}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={!dataset}
        className="field"
      >
        <option value="">{optional ? "None" : "Choose a column"}</option>
        {options.map((column) => (
          <option key={column} value={column}>
            {column}
          </option>
        ))}
      </select>
    </div>
  );

  const many = (
    legend: string,
    name: string,
    values: string[],
    options: string[],
    onChange: (next: string[]) => void,
    note?: string,
  ) => (
    <fieldset disabled={!dataset} className="choices">
      <legend>{legend}</legend>
      {note ? <p className="mb-2 max-w-measure text-note text-ink-soft">{note}</p> : null}
      <div className="choice-list">
        {options.map((column) => (
          <label key={column} className="check value">
            <input
              type="checkbox"
              name={name}
              value={column}
              checked={values.includes(column)}
              onChange={(event) => onChange(toggle(values, column, event.target.checked))}
            />
            {column}
          </label>
        ))}
      </div>
    </fieldset>
  );

  return (
    <form onSubmit={submit} className="flex max-w-measure flex-col gap-section">
      <section aria-labelledby="dataset-heading">
        <h2 id="dataset-heading" className="title-section">
          Dataset
        </h2>
        <div className="flex flex-col gap-5">
          <div className={FIELD_BOX}>
            <label htmlFor="file" className="label">
              Upload a CSV
            </label>
            <input
              id="file"
              type="file"
              accept=".csv,text/csv"
              onChange={upload}
              disabled={uploading}
              className="field cursor-pointer py-1.5 file:mr-3 file:cursor-pointer file:border-0 file:bg-transparent file:p-0 file:font-sans file:text-note file:font-medium file:text-accent"
            />
          </div>
          {datasets.length > 0 ? (
            <div className={FIELD_BOX}>
              <label htmlFor="existing" className="label">
                or use an earlier upload
              </label>
              <select id="existing" defaultValue="" onChange={pickExisting} className="field">
                <option value="">Choose a dataset</option>
                {datasets.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.filename} ({formatBytes(item.n_bytes)})
                  </option>
                ))}
              </select>
            </div>
          ) : null}
          {uploading ? (
            <p role="status" className="text-note text-ink-soft">
              Uploading.
            </p>
          ) : null}
          {dataset ? (
            <p data-dataset={dataset.id} className="value break-all">
              {dataset.filename}: {dataset.columns.length} columns
            </p>
          ) : null}
          {datasetError ? (
            <p role="alert" className="engine engine-alert">
              {datasetError}
            </p>
          ) : null}
        </div>
      </section>

      <section aria-labelledby="schema-heading">
        <h2 id="schema-heading" className="title-section">
          Schema
        </h2>
        <div className="flex flex-col gap-6">
          <div className={FIELD_GRID}>
            {single("unit", "Unit", form.unit, columnsFor("unit", form, columns), (column) =>
              setForm((current) => withRole(current, "unit", column)),
            )}
            {single("treatment", "Treatment", form.treatment, columnsFor("treatment", form, columns), (column) =>
              setForm((current) => withRole(current, "treatment", column)),
            )}
            {single("outcome", "Outcome", form.outcome, columnsFor("outcome", form, columns), (column) =>
              setForm((current) => withRole(current, "outcome", column)),
            )}
            <div className={FIELD_BOX}>
              <label htmlFor="outcome-type" className="label">
                Outcome type
              </label>
              <select
                id="outcome-type"
                value={form.outcomeType}
                onChange={(event) =>
                  setForm((current) => ({ ...current, outcomeType: event.target.value as OutcomeType | "" }))
                }
                className="field"
              >
                <option value="">Choose a type</option>
                {OUTCOME_TYPES.map((type) => (
                  <option key={type} value={type}>
                    {type}
                  </option>
                ))}
              </select>
            </div>
          </div>
          <div className={FIELD_BOX}>
            <label htmlFor="provenance" className="label">
              Assignment provenance
            </label>
            <select
              id="provenance"
              value={form.provenance}
              onChange={(event) =>
                setForm((current) => ({ ...current, provenance: event.target.value as Provenance }))
              }
              className="field sm:max-w-[calc(50%-0.75rem)]"
            >
              {PROVENANCES.map((provenance) => (
                <option key={provenance} value={provenance}>
                  {provenance}
                </option>
              ))}
            </select>
            <p className="text-note text-ink-soft">How treatment came to be assigned. It is a declaration the data cannot check, and it caps the validity band.</p>
          </div>
          {many(
            "Covariates",
            "covariates",
            form.covariates,
            columnsFor("covariates", form, columns),
            (next) => setForm((current) => ({ ...current, covariates: next })),
            form.covariates.length === 0
              ? "None chosen: the analysis will be told there are no covariates to adjust for."
              : undefined,
          )}
          <div>
            <details className="disclosure">
              <summary>Panel</summary>
              <div className={FIELD_GRID}>
                {single("time", "Time", form.time, columns, (column) =>
                  setForm((current) => ({ ...current, time: column })), true)}
                {single("cohort", "Cohort", form.cohort, columns, (column) =>
                  setForm((current) => ({ ...current, cohort: column })), true)}
              </div>
            </details>
            <details className="disclosure">
              <summary>Advanced</summary>
              <div className="flex flex-col gap-5">
                <div className={FIELD_GRID}>
                  {single("negative-control", "Negative-control outcome", form.negativeControl, columns, (column) =>
                    setForm((current) => ({ ...current, negativeControl: column })), true)}
                </div>
                {many("Auxiliary columns", "auxiliary", form.auxiliary, columns, (next) =>
                  setForm((current) => ({ ...current, auxiliary: next })))}
              </div>
            </details>
          </div>
        </div>
      </section>

      <section aria-labelledby="estimators-heading">
        <h2 id="estimators-heading" className="title-section">
          Estimators
        </h2>
        <div className="flex flex-col gap-5">
          <div>
            <label className="check">
              <input type="radio" name="mode" checked={routerChooses} onChange={() => setRouterChooses(true)} />
              Let the router choose
            </label>
            <p className="ml-6.5 max-w-measure text-note text-ink-soft">
              Every estimator the router finds eligible is fitted. On panel data that includes
              estimators sampled with MCMC, which can take minutes. An analysis is stopped after 30
              minutes.
            </p>
          </div>
          <div>
            <label className="check">
              <input type="radio" name="mode" checked={!routerChooses} onChange={() => setRouterChooses(false)} />
              Choose estimators
            </label>
            {routerChooses ? null : (
              <div className="mt-2 flex flex-col gap-5">
                {FAMILIES.map((family) => (
                  <fieldset key={family} className="choices">
                    <legend>{FAMILY_LABELS[family]}</legend>
                    <div className="choice-list">
                      {estimators
                        .filter((entry) => entry.family === family)
                        .map((entry) => (
                          <label key={entry.name} className="check value">
                            <input
                              type="checkbox"
                              name="estimators"
                              value={entry.name}
                              checked={picked.includes(entry.name)}
                              onChange={(event) => setPicked(toggle(picked, entry.name, event.target.checked))}
                            />
                            {entry.name}
                          </label>
                        ))}
                    </div>
                  </fieldset>
                ))}
                <p className="text-note text-ink-soft">The groups are for finding an estimator. Whether one can run on this dataset is decided when the analysis runs, and shown in its routing table.</p>
              </div>
            )}
          </div>
          <div className={`${FIELD_BOX} sm:max-w-[calc(50%-0.75rem)]`}>
            <label htmlFor="seed" className="label">
              Random seed
            </label>
            <input
              id="seed"
              inputMode="numeric"
              value={seedText}
              onChange={(event) => setSeedText(event.target.value)}
              className="field"
            />
          </div>
          {seed === null ? (
            <p role="alert" className="engine engine-alert">
              The seed is a whole number from 0 to 2147483647.
            </p>
          ) : null}
        </div>
      </section>

      <div className="flex flex-col items-start gap-4">
        {dataset && missing.length > 0 ? (
          <p className="text-note text-ink-soft">Still needed: {missing.join(", ")}.</p>
        ) : null}
        {nonePicked ? (
          <p className="text-note text-ink-soft">Choose at least one estimator, or let the router choose.</p>
        ) : null}
        {submitError ? (
          <p role="alert" className="engine engine-alert">
            {submitError}
          </p>
        ) : null}
        <button
          type="submit"
          disabled={blocked || submitting}
          className={`button button-primary ${submitting ? "" : "disabled:cursor-not-allowed"}`}
        >
          {submitting ? "Submitting" : "Run the analysis"}
        </button>
      </div>
    </form>
  );
}
