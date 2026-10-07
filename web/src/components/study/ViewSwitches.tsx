import Link from "next/link";

import { levelFor } from "@/lib/study/matrix";
import type { Level } from "@/lib/study/matrix";
import type { View } from "@/lib/study/query";

type Offered = { family: string; metrics: string[]; levels: Level[] };

type Choice = { label: string; href: string; current: boolean };

const LEVEL_LABELS: Record<Level, string> = {
  scenario: "By scenario",
  scenario_by_size: "By scenario and size",
};

const SHAPE_CHOICES: [View["shape"], string][] = [
  ["all", "All"],
  ["cross_section", "Cross-sectional"],
  ["panel", "Panel"],
  ["donor_pool", "Donor pool"],
];

const ESTIMATOR_CHOICES: [View["estimatorFamily"], string][] = [
  ["all", "All"],
  ["cross_sectional", "Cross-sectional"],
  ["panel", "Panel"],
  ["donor_pool", "Donor pool"],
];

const DISPLAY_CHOICES: [View["display"], string][] = [
  ["heatmap", "Heatmap"],
  ["table", "Table"],
];

/**
 * The view's switches, every one a link to the view it leads to (`href` is
 * `viewHref` of that view), so a view is a URL and the back button steps
 * through them with no script. The current choice of each group is marked
 * `aria-current`. A switch that changes which columns the matrix has drops the
 * sort, which names one of them.
 */
export function ViewSwitches({
  view,
  offered,
  href,
}: {
  view: View;
  offered: readonly Offered[];
  href: (view: View) => string;
}) {
  const current = offered.find((entry) => entry.family === view.family);
  const unsorted = { sort: null, direction: "asc" } as const;

  const families: Choice[] = offered.map((entry) => ({
    label: entry.family,
    // A family's own first metric; a level it does not hold falls back to one it does.
    href: href({
      ...view,
      family: entry.family,
      metric: entry.metrics[0] ?? view.metric,
      level: levelFor(entry.levels, view.level),
    }),
    current: entry.family === view.family,
  }));

  const metrics: Choice[] = (current?.metrics ?? []).map((metric) => ({
    label: metric,
    href: href({ ...view, metric }),
    current: metric === view.metric,
  }));

  const levels: Choice[] = (current?.levels ?? []).map((level) => ({
    label: LEVEL_LABELS[level],
    href: href({ ...view, level, ...unsorted }),
    current: level === view.level,
  }));

  const shapes: Choice[] = SHAPE_CHOICES.map(([shape, label]) => ({
    label,
    href: href({ ...view, shape, ...unsorted }),
    current: shape === view.shape,
  }));

  const estimators: Choice[] = ESTIMATOR_CHOICES.map(([estimatorFamily, label]) => ({
    label,
    href: href({ ...view, estimatorFamily }),
    current: estimatorFamily === view.estimatorFamily,
  }));

  const displays: Choice[] = DISPLAY_CHOICES.map(([display, label]) => ({
    label,
    href: href({ ...view, display }),
    current: display === view.display,
  }));

  return (
    <div className="flex flex-col gap-4">
      <SwitchGroup label="Metric family" choices={families} names />
      <SwitchGroup label="Metric" choices={metrics} names />
      <div className="flex flex-wrap gap-x-8 gap-y-4">
        <SwitchGroup label="Level" choices={levels} />
        <SwitchGroup label="Scenarios" choices={shapes} />
        <SwitchGroup label="Estimators" choices={estimators} />
        <SwitchGroup label="Display" choices={displays} />
      </div>
    </div>
  );
}

/** `names`: the choices are the study's own names, so they are set as values. */
function SwitchGroup({ label, choices, names = false }: { label: string; choices: Choice[]; names?: boolean }) {
  return (
    <div role="group" aria-label={label} className="min-w-0">
      <p className="label mb-1.5" aria-hidden="true">
        {label}
      </p>
      <div className="flex flex-wrap gap-2">
        {choices.map((choice) => (
          <Link
            key={choice.label}
            href={choice.href}
            scroll={false}
            aria-current={choice.current ? "true" : undefined}
            className={names ? "switch value" : "switch"}
          >
            {choice.label}
          </Link>
        ))}
      </div>
    </div>
  );
}
