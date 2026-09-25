// The frontend's single mutable state object, plus shared constants. Field names correspond
// one-to-one with backend models (DocumentSummary / ComparisonReport / LaneExtraction / ParsedArtifact / Job).

export const LANES = ["mineru", "paddleocr_vl"];
export const LANE_LABEL = { mineru: "MinerU", paddleocr_vl: "PaddleOCR-VL" };
// One vocabulary for the comparison outcomes: filter chips, KPI tiles, the library tally and the status cell.
export const STATUS = {
  agree: { label: "一致", note: "两路一致，直接接受" },
  conflict: { label: "冲突", note: "两路数值冲突，待裁决" },
  ambiguous: { label: "不确定", note: "无法判断，待裁决" },
  missing: { label: "缺失", note: "只有一路抽到" },
};
export const STATUS_ORDER = ["agree", "conflict", "ambiguous", "missing"];
export const STAGE_LABEL = {
  "parse:mineru": "解析 MinerU",
  "parse:paddleocr_vl": "解析 PaddleOCR-VL",
  figures: "读图",
  "extract:mineru": "抽取 MinerU",
  "extract:paddleocr_vl": "抽取 PaddleOCR-VL",
  compare: "对齐比较",
  export: "导出",
};
// A stage's state is a word and a glyph as well as a colour, so it never rests on colour alone.
export const STAGE_STATUS = {
  pending: { label: "未开始", glyph: "○" },
  running: { label: "进行中", glyph: "…" },
  done: { label: "已完成", glyph: "✓" },
  failed: { label: "失败", glyph: "✕" },
  skipped: { label: "已跳过", glyph: "–" },
};
export const JOB_STATUS_LABEL = { queued: "排队中", running: "处理中", done: "已完成", failed: "失败" };
const ACTIVE_JOB_STATUS = new Set(["queued", "running"]);

export const state = {
  // Bumped by the router whenever the view changes to another document or to home (router.js). Every
  // async load, poll and finish handler notes it before its first await and draws only while it is still
  // the same: a late response then can never paint over whatever the reader navigated to since.
  generation: 0,
  docs: [],        // document library list (DocumentSummary[])
  activeDocs: new Set(), // document_ids with a queued or running job (from GET /api/jobs)
  current: null,   // the open document_id (16 chars)
  summary: null,   // the current document's DocumentSummary
  report: null,    // ComparisonReport | null
  dataset: null,   // the consolidated per-sample table (DocumentDataset.as_dict) | null
  figures: null,   // chart readings from the figures stage ({ stale, orphaned, rows })
  corpus: null,    // the home view's library-wide table ({ fields, rows }) | null
  lanes: {},       // { mineru: LaneExtraction | null, paddleocr_vl: ... }
  artifacts: {},   // { mineru: ParsedArtifact | null, paddleocr_vl: ... }
  job: null,       // the current document's most recent job snapshot (this process only)
  filter: null,    // status filter for the facts table
  selectedFact: null, // index into report.comparisons of the fact in the URL, even while a filter hides it
  viewer: null,    // the PageViewer instance
};

export const isCurrent = (generation) => generation === state.generation;

// Only a job belonging to the current document is used to draw progress; after switching
// documents, a stale job snapshot must not carry over onto the new one
export const currentJob = () => (state.job && state.job.document_id === state.current ? state.job : null);
export const isActive = (job) => Boolean(job) && ACTIVE_JOB_STATUS.has(job.status);

// slots in the document view template
export const slot = (name, root = document.getElementById("document-view")) => root.querySelector(`[data-slot="${name}"]`);

// Why a processed paper has no samples. The extraction keeps the inventory's "this paper deposits no TCO
// film of its own" verdict only as the reason it skipped every sample-level question (extract.py), so that
// reason is what is looked for; any other empty lane just found no samples.
const NO_FILM_REASON = "deposits no TCO film";
export function noSamplesReason() {
  const lanes = LANES.map((lane) => state.lanes[lane]).filter(Boolean);
  const noFilm =
    lanes.length > 0 &&
    lanes.every((lane) => !lane.samples?.length && (lane.dropped ?? []).some((reason) => reason.includes(NO_FILM_REASON)));
  return noFilm ? "该论文没有自己沉积的 TCO 膜，所以没有样品级数据。" : "未识别到样品：两路抽取都没有给出样品。";
}
