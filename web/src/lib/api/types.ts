import type { components } from "./schema";

type Schemas = components["schemas"];

export type Me = Schemas["Me"];
export type EstimatorEntry = Schemas["EstimatorEntry"];
export type Family = EstimatorEntry["family"];
export type DatasetUploaded = Schemas["DatasetUploaded"];
export type DatasetItem = Schemas["DatasetItem"];
export type DatasetDetail = Schemas["DatasetDetail"];
export type DatasetPage = Schemas["DatasetPage"];
export type AnalysisCreated = Schemas["AnalysisCreated"];
export type AnalysisItem = Schemas["AnalysisItem"];
export type AnalysisPage = Schemas["AnalysisPage"];
export type AnalysisDetail = Schemas["AnalysisDetail"];
export type Status = AnalysisDetail["status"];
export type Estimate = Schemas["Estimate"];
export type EstimateSummary = Schemas["EstimateSummary"];
export type RoutingDecision = Schemas["RoutingDecision"];
export type ComparisonGroup = Schemas["ComparisonGroup"];
export type Summary = Schemas["Summary"];
export type AnalysisRequest = Schemas["AnalysisRequest"];
export type InterpretationDetail = Schemas["InterpretationDetail"];
export type InterpretationStatus = InterpretationDetail["status"];
export type InterpretationOption = Schemas["InterpretationOptionOut"];
/** An entry of the server's list of models and efforts, as a request names it. */
export type InterpretationChoice = { model: string; effort: string };
