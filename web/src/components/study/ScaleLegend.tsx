import type { CSSProperties } from "react";

import { formatNumber } from "@/lib/format";
import { legendSteps } from "@/lib/study/scale";
import type { Scale } from "@/lib/study/scale";

/** An end's value, behind `≤` or `≥` only where values in view lie beyond it. */
function endLabel(value: number, beyond: boolean, sign: "≤" | "≥"): string {
  return beyond ? `${sign} ${formatNumber(value)}` : formatNumber(value);
}

/**
 * The scale in words, beside a strip of its steps. Every word comes from the
 * scale it is given, and the strip is drawn with the cells' own colour rule
 * (`heat`, `--pos`), over only the positions a value can take (`legendSteps`),
 * so neither can drift from the cells. Nothing is said when the scale is flat:
 * there is nothing to compare.
 */
export function ScaleLegend({ scale }: { scale: Scale }) {
  if (scale.kind === "flat") return null;
  const diverging = scale.kind === "diverging";
  const steps = legendSteps(scale);
  const low = diverging ? endLabel(scale.lowEnd, scale.lowBeyond, "≤") : endLabel(scale.low, scale.lowBeyond === true, "≤");
  const high = diverging ? endLabel(scale.highEnd, scale.highBeyond, "≥") : endLabel(scale.high, scale.highBeyond === true, "≥");
  const words = diverging ? `diverging about ${scale.centreLabel}` : `low to high: ${low} to ${high}`;
  return (
    <p className="label mb-3 flex flex-wrap items-center gap-x-3 gap-y-1">
      <span>{words}</span>
      <span className="inline-flex items-center gap-2">
        {diverging ? <span className="value">{low}</span> : null}
        <span className="scale-strip" aria-hidden="true">
          {steps.map((step, index) => (
            <span
              key={index}
              className="scale-step heat"
              data-scale={scale.kind}
              style={{ "--pos": step.toFixed(4) } as CSSProperties}
            />
          ))}
        </span>
        {diverging ? <span className="value">{high}</span> : null}
      </span>
    </p>
  );
}
