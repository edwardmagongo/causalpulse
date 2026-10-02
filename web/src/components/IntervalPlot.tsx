import { formatInterval, formatLevel, formatNumber } from "@/lib/format";
import { position, showsZero, type PlotPanel } from "@/lib/plot";

const WIDTH = 640;
const LABEL = 190;
const RIGHT = 16;
const ROW = 36;
const TOP = 8;
const AXIS = 26;

/**
 * Point and interval for each method, one panel per comparison group.
 *
 * Methods share an axis only inside a panel. The estimates table beside it
 * carries the same numbers as text, which is this plot's accessible form.
 */
export function IntervalPlot({ panels }: { panels: PlotPanel[] }) {
  if (panels.length === 0) return null;
  return (
    <section aria-labelledby="plot-heading">
      <h2 id="plot-heading">Points and intervals</h2>
      {panels.map((panel) => {
        const height = TOP + panel.rows.length * ROW + AXIS;
        const x = (value: number) =>
          LABEL + position(value, panel.domain) * (WIDTH - LABEL - RIGHT);
        const axisY = TOP + panel.rows.length * ROW;
        return (
          <figure key={`${panel.estimand}/${panel.scale}`}>
            <figcaption>
              {panel.estimand}, {panel.scale} scale
            </figcaption>
            <svg
              viewBox={`0 0 ${WIDTH} ${height}`}
              role="img"
              aria-label={`Point and interval for ${panel.rows
                .map((row) => row.method)
                .join(", ")}: ${panel.estimand}, ${panel.scale} scale`}
            >
              {showsZero(panel.domain) ? (
                <line data-zero="true" x1={x(0)} x2={x(0)} y1={TOP} y2={axisY} stroke="currentColor" strokeDasharray="3 3" opacity={0.5} />
              ) : null}
              {panel.rows.map((row, index) => {
                const y = TOP + index * ROW + ROW / 2;
                return (
                  <g key={row.method} data-method={row.method}>
                    <title>
                      {`${row.method}: ${formatNumber(row.point)}, ${formatLevel(row.level)} ${
                        row.basis
                      } interval ${formatInterval(row.lower, row.upper)}`}
                    </title>
                    <text x={LABEL - 10} y={y - 7} textAnchor="end" dominantBaseline="middle" fontSize={12} fill="currentColor">
                      {row.method}
                    </text>
                    <text
                      data-basis={row.basis}
                      x={LABEL - 10}
                      y={y + 8}
                      textAnchor="end"
                      dominantBaseline="middle"
                      fontSize={10}
                      fill="currentColor"
                    >
                      {`${formatLevel(row.level)} ${row.basis} interval`}
                    </text>
                    <line x1={x(row.lower)} x2={x(row.upper)} y1={y} y2={y} stroke="currentColor" strokeWidth={2} />
                    <circle cx={x(row.point)} cy={y} r={4} fill="currentColor" />
                  </g>
                );
              })}
              <line x1={LABEL} x2={WIDTH - RIGHT} y1={axisY} y2={axisY} stroke="currentColor" />
              <text x={LABEL} y={axisY + 16} fontSize={11} fill="currentColor">
                {formatNumber(panel.domain[0])}
              </text>
              <text x={WIDTH - RIGHT} y={axisY + 16} textAnchor="end" fontSize={11} fill="currentColor">
                {formatNumber(panel.domain[1])}
              </text>
            </svg>
          </figure>
        );
      })}
    </section>
  );
}
