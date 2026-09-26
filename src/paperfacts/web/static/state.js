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
  profile: null,   // the served domain profile's title, UI copy, groups and fields (GET /api/profile)
};

export const isCurrent = (generation) => generation === state.generation;

// ui_copy.UiCopy's domain-free defaults, mirrored here (tests/test_web_copy.py holds the two equal): until the
// profile has loaded, or when it never does, the page still names things, just generically.
const GENERIC_UI_COPY = {
  paper_level_label_zh: "论文级",
  paper_level_short_zh: "论文级",
  entity_label_zh: "样品",
  no_samples_message_zh: "该论文没有范围内的样品，所以没有样品级数据。",
};

// One line of the profile's display copy (ui_copy.UiCopy). Every domain word on the page comes through here, so
// the page names a paper-level record or a sample the way the served profile does.
export const uiCopy = (key) => state.profile?.ui?.[key] ?? GENERIC_UI_COPY[key] ?? "";

// Static markup names the entity through an empty `<span data-ui="key">`; this fills every one under `root`.
// The document template is filled on every clone, the page itself once at start and again when the profile lands.
export function applyUiCopy(root) {
  for (const element of root.querySelectorAll("[data-ui]")) element.textContent = uiCopy(element.dataset.ui);
}

// ---------- entity types ----------
//
// A profile may declare several kinds of sample (profile.entities, the primary first): each has its own rows,
// records and matching, and the page groups by them under their own labels. A profile without entity types -- and a
// page whose profile has not loaded -- has one group, `name: null`, holding everything; nothing on the page names it,
// so such a page looks exactly as it did before entity types existed.

// The name a lane sample, a dataset row or a comparison scope has when it carries none: the implicit entity's.
const IMPLICIT_ENTITY = "sample";

export function entityGroups() {
  const declared = state.profile?.entities ?? [];
  if (declared.length < 2) return [{ name: null, label: uiCopy("entity_label_zh") }];
  return declared.map((entity) => ({ name: entity.name, label: entity.label_zh || entity.name }));
}

// Whether a lane sample, a dataset row or a quality row is one of `group`'s. Only a sample-level field column is
// asked (a paper-level one belongs to no entity).
export const inEntity = (group, item) => group.name === null || entityOf(item) === group.name;
export const entityOf = (item) => item?.entity ?? IMPLICIT_ENTITY;

// The label of an entity by its name, or "" when the page has only the one group (and so names none).
export function entityLabel(name) {
  const groups = entityGroups();
  if (groups.length < 2) return "";
  return groups.find((group) => group.name === name)?.label ?? name;
}

// Only a job belonging to the current document is used to draw progress; after switching
// documents, a stale job snapshot must not carry over onto the new one
export const currentJob = () => (state.job && state.job.document_id === state.current ? state.job : null);
export const isActive = (job) => Boolean(job) && ACTIVE_JOB_STATUS.has(job.status);

// slots in the document view template
export const slot = (name, root = document.getElementById("document-view")) => root.querySelector(`[data-slot="${name}"]`);

// Why a processed paper has no samples. Each lane carries the inventory's "this paper reports no in-scope
// sample of its own" verdict as `no_samples` (extract.py), and the profile words it; any
// other empty lane just found no samples.
export function noSamplesReason() {
  const lanes = LANES.map((lane) => state.lanes[lane]).filter(Boolean);
  const noneInScope = lanes.length > 0 && lanes.every((lane) => !lane.samples?.length && lane.no_samples === true);
  return noneInScope ? uiCopy("no_samples_message_zh") : `未识别到${uiCopy("entity_label_zh")}：两路抽取都没有给出${uiCopy("entity_label_zh")}。`;
}
