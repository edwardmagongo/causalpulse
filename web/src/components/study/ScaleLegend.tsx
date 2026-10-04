import { formatNumber } from "@/lib/format";
import type { Scale } from "@/lib/study/scale";

/** The coverage metrics and the false-positive rate sit about a level the study sets out to reach. */
function isNominal(metric: string): boolean {
  return metric.startsWith("coverage_") || metric === "false_positive_rate";
}

/**
 * The scale in words, beside a strip of its steps. The strip is drawn with the
 * cells' own colour rule (`heat`, `--pos`), so it cannot drift from the cells.
 * Nothing is said when the scale is flat: there is nothing to compare.
 */
export function ScaleLegend({ scale, metric }: { scale: Scale; metric: string }) {
  if (scale.kind === "flat") return null;
  const diverging = scale.kind === "diverging";
  const steps = diverging ? [-1, -0.5, 0, 0.5, 1] : [0, 0.25, 0.5, 0.75, 1];
  const words = diverging
    ? `diverging about ${scale.centreLabel}${isNominal(metric) ? ", the nominal level" : ""}`
    : `low to high: ${formatNumber(scale.low)} to ${formatNumber(scale.high)}`;
  return (
    <p className="label mb-3 flex flex-wrap items-center gap-x-3 gap-y-1">
      <span>{words}</span>
      <span className="inline-flex items-center gap-2">
        {diverging ? <span className="value">{formatNumber(scale.centre - scale.halfWidth)}</span> : null}
        <span className="scale-strip" aria-hidden="true">
          {steps.map((step) => (
            <span
              key={step}
              className="scale-step heat"
              data-scale={scale.kind}
              style={{ "--pos": step } as React.CSSProperties}
            />
          ))}
        </span>
        {diverging ? <span className="value">{formatNumber(scale.centre + scale.halfWidth)}</span> : null}
      </span>
    </p>
  );
}
