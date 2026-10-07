/**
 * The `/study` view as a query string. The URL is the view's whole state, and
 * everything in it is parsed against what the study offers: a value that is not
 * recognised is replaced by its default, never echoed, so no query text reaches
 * the page unparsed.
 */
import { levelFor, matrixOf, offeredFamilies } from "./matrix";
import type { Level } from "./matrix";
import { isEstimatorFamily, isShape } from "./types";
import type { EstimatorFamily, Shape, StudyBundle } from "./types";

export type View = {
  family: string;
  metric: string;
  level: Level;
  shape: Shape | "all";
  estimatorFamily: EstimatorFamily | "all";
  display: "heatmap" | "table";
  sort: string | null;
  direction: "asc" | "desc";
};

type Params = Record<string, string | string[] | undefined>;

const DEFAULT_FAMILY = "accuracy";
const DEFAULT_METRIC = "bias";

/** A key's value, as one string: the first of a repeated key. */
function first(params: Params, key: string): string | undefined {
  const value = params[key];
  return Array.isArray(value) ? value[0] : value;
}

function metricsOf(study: StudyBundle, family: string): string[] | undefined {
  return offeredFamilies(study).find((offered) => offered.family === family)?.metrics;
}

function levelsOf(study: StudyBundle, family: string): Level[] {
  return offeredFamilies(study).find((offered) => offered.family === family)?.levels ?? [];
}

export function defaultView(study: StudyBundle): View {
  const offered = offeredFamilies(study);
  const family = offered.some((o) => o.family === DEFAULT_FAMILY) ? DEFAULT_FAMILY : (offered[0]?.family ?? DEFAULT_FAMILY);
  const metrics = metricsOf(study, family) ?? [];
  return {
    family,
    metric: metrics.includes(DEFAULT_METRIC) ? DEFAULT_METRIC : (metrics[0] ?? DEFAULT_METRIC),
    level: "scenario",
    shape: "all",
    estimatorFamily: "all",
    display: "heatmap",
    sort: null,
    direction: "asc",
  };
}

export function parseView(study: StudyBundle, params: Params): View {
  const base = defaultView(study);

  const requestedFamily = first(params, "family");
  const family =
    requestedFamily !== undefined && metricsOf(study, requestedFamily) !== undefined ? requestedFamily : base.family;
  const metrics = metricsOf(study, family) ?? [];

  const requestedMetric = first(params, "metric");
  const metric =
    requestedMetric !== undefined && metrics.includes(requestedMetric)
      ? requestedMetric
      : metrics.includes(base.metric)
        ? base.metric
        : (metrics[0] ?? base.metric);

  const requestedLevel = first(params, "level");
  const wantedLevel: Level =
    requestedLevel === "scenario" || requestedLevel === "scenario_by_size" ? requestedLevel : base.level;
  // A level the chosen family has no table for would show an empty matrix: take one it offers.
  const level = levelFor(levelsOf(study, family), wantedLevel);

  const requestedShape = first(params, "shape");
  const shape: Shape | "all" = requestedShape !== undefined && isShape(requestedShape) ? requestedShape : "all";

  const requestedEstimators = first(params, "estimators");
  const estimatorFamily: EstimatorFamily | "all" =
    requestedEstimators !== undefined && isEstimatorFamily(requestedEstimators) ? requestedEstimators : "all";

  const requestedDisplay = first(params, "display");
  const display: View["display"] = requestedDisplay === "table" ? "table" : "heatmap";

  const requestedDirection = first(params, "dir");
  const direction: View["direction"] = requestedDirection === "desc" ? "desc" : "asc";

  // A sort names the estimator column or a column of the matrix this view shows.
  const requestedSort = first(params, "sort");
  let sort: string | null = null;
  if (requestedSort !== undefined) {
    const columns = matrixOf(study, { family, metric, level, shape, estimatorFamily }).columns;
    if (requestedSort === "estimator" || columns.some((column) => column.key === requestedSort)) {
      sort = requestedSort;
    }
  }

  return { family, metric, level, shape, estimatorFamily, display, sort, direction };
}

/** `/study` and only the keys that differ from the default view. */
export function viewHref(study: StudyBundle, view: View): string {
  const base = defaultView(study);
  const query = new URLSearchParams();
  if (view.family !== base.family) query.set("family", view.family);
  if (view.metric !== base.metric) query.set("metric", view.metric);
  if (view.level !== base.level) query.set("level", view.level);
  if (view.shape !== base.shape) query.set("shape", view.shape);
  if (view.estimatorFamily !== base.estimatorFamily) query.set("estimators", view.estimatorFamily);
  if (view.display !== base.display) query.set("display", view.display);
  if (view.sort !== base.sort && view.sort !== null) query.set("sort", view.sort);
  if (view.direction !== base.direction) query.set("dir", view.direction);
  const text = query.toString();
  return text === "" ? "/study" : `/study?${text}`;
}
