import { DASH, formatNumber } from "@/lib/format";
import type { ReportTableRow, ReportTables } from "@/lib/study/types";

/**
 * The report's own two tables as it prints them, each under its title, with the
 * report's notes where it puts them: the coverage note directly after the table
 * it describes (the one with an `interval` column), the exploratory note after
 * the tables. Plain tables, not coloured, and every string is the bundle's own.
 * A record's own keys are alphabetical, so the columns are declared here: the
 * method, the interval if the rows have one, then the scenarios in `scenarios`
 * order. A note that is empty is not drawn.
 */
export function PublishedTables({
  reportTables,
  scenarios,
}: {
  reportTables: ReportTables;
  scenarios: readonly string[];
}) {
  const entries = Object.entries(reportTables.tables);
  const coverageTitle = entries.find(([, rows]) => hasInterval(rows))?.[0];
  // A coverage note with no table to sit beside is still the bundle's, and is not dropped.
  const orphanNote = coverageTitle === undefined ? reportTables.coverage_note : "";
  return (
    <section aria-labelledby="published" className="mt-10">
      <h3 id="published" className="title-group mb-4">
        As published
      </h3>
      <div className="flex flex-col gap-8">
        {entries.map(([title, rows]) => (
          <div key={title}>
            <h4 className="label mb-2">{title}</h4>
            <PublishedTable title={title} rows={rows} scenarios={scenarios} />
            {title === coverageTitle ? <Note text={reportTables.coverage_note} /> : null}
          </div>
        ))}
      </div>
      <Note text={orphanNote} />
      <Note text={reportTables.exploratory_note} />
    </section>
  );
}

function hasInterval(rows: readonly ReportTableRow[]): boolean {
  return rows.some((row) => row.interval !== undefined);
}

function Note({ text }: { text: string }) {
  return text === "" ? null : <p className="engine mt-4">{text}</p>;
}

function PublishedTable({
  title,
  rows,
  scenarios,
}: {
  title: string;
  rows: readonly ReportTableRow[];
  scenarios: readonly string[];
}) {
  const withInterval = hasInterval(rows);
  return (
    <div className="table-scroll">
      <table className="data-table">
        <caption className="sr-only">{title}</caption>
        <thead>
          <tr>
            <th scope="col">method</th>
            {withInterval ? <th scope="col">interval</th> : null}
            {scenarios.map((scenario) => (
              <th key={scenario} scope="col" className="num value">
                {scenario}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.method}>
              <th scope="row">{row.method}</th>
              {withInterval ? <td>{row.interval === undefined || row.interval === "" ? DASH : row.interval}</td> : null}
              {scenarios.map((scenario) => {
                const cell = row[scenario];
                return (
                  <td key={scenario} className="num">
                    {formatNumber(typeof cell === "number" ? cell : null)}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
