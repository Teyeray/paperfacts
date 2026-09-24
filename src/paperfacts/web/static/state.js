// The frontend's single mutable state object, plus shared constants. Field names correspond
// one-to-one with backend models (DocumentSummary / ComparisonReport / LaneExtraction / ParsedArtifact / Job).

export const LANES = ["mineru", "paddleocr_vl"];
export const LANE_LABEL = { mineru: "MinerU", paddleocr_vl: "PaddleOCR-VL" };
export const STATUS_LABEL = { agree: "AGREE", conflict: "CONFLICT", ambiguous: "AMBIGUOUS", missing: "MISSING" };
export const STATUS_NOTE = { agree: "两路一致，直接接受", conflict: "两路数值冲突，待裁决", ambiguous: "无法判断，待裁决", missing: "只有一路抽到" };
export const STAGE_LABEL = {
  "parse:mineru": "解析 MinerU",
  "parse:paddleocr_vl": "解析 PaddleOCR-VL",
  "extract:mineru": "抽取 MinerU",
  "extract:paddleocr_vl": "抽取 PaddleOCR-VL",
  figures: "读图",
  compare: "对齐比较",
};
const ACTIVE_JOB_STATUS = new Set(["queued", "running"]);

export const state = {
  docs: [],        // document library list (DocumentSummary[])
  activeDocs: new Set(), // document_ids with a queued or running job (from GET /api/jobs)
  current: null,   // the open document_id (16 chars)
  summary: null,   // the current document's DocumentSummary
  report: null,    // ComparisonReport | null
  dataset: null,   // the consolidated per-sample table (DocumentDataset.as_dict) | null
  figures: null,   // chart readings from the figures stage ({ stale, orphaned, rows }) | null
  corpus: null,    // the home view's library-wide table ({ fields, rows }) | null
  lanes: {},       // { mineru: LaneExtraction | null, paddleocr_vl: ... }
  artifacts: {},   // { mineru: ParsedArtifact | null, paddleocr_vl: ... }
  job: null,       // the current document's most recent job snapshot (this process only)
  filter: null,    // status filter for the facts table
  viewer: null,    // the PageViewer instance
};

// Only a job belonging to the current document is used to draw progress; after switching
// documents, a stale job snapshot must not carry over onto the new one
export const currentJob = () => (state.job && state.job.document_id === state.current ? state.job : null);
export const isActive = (job) => Boolean(job) && ACTIVE_JOB_STATUS.has(job.status);

// slots in the document view template
export const slot = (name, root = document.getElementById("document-view")) => root.querySelector(`[data-slot="${name}"]`);
