// The read-only profile page: what a domain profile asks and how it judges -- its entities, groups, fields, declared
// units, retrieval words, and the prompts it renders. Nothing here edits a profile.
//
// `renderDefinition` draws any definition object (GET /api/profiles/<name>: profile_view.profile_definition) into a
// container, so a page that checks a pasted profile draws exactly the page that profile would get. Every piece of
// profile text lands through textContent (or escapeHtml where a table column is spliced as markup): a profile is data,
// and a label holding markup reads as that markup, never as an element.

import { escapeHtml } from "./html.js";
import { copyButton, copyTable } from "./tsv.js";

// What a FieldSpec attribute is called on the page. The table's columns come from the definition itself (every
// attribute the server sends, the ones below first), so an attribute added later shows up under its own name.
const ATTRIBUTE_LABEL = {
  name: "字段",
  label: "显示名",
  kind: "类型",
  level: "层级",
  entity: "实体",
  canonical_unit: "单位",
  categories: "类别",
  cardinality: "取值个数",
  range_policy: "范围取法",
  references: "引用",
  condition_rule: "条件规则",
  valid_range: "合理范围",
  group: "分组",
  rel_tol: "相对容差",
  abs_tol: "绝对容差",
  keywords: "检索关键词",
  condition_hint: "条件提示",
  condition_preference: "条件优先",
  bare_number: "无单位数字",
  after_clause: "后置从句",
  prompt_categories: "提示中的类别",
  figure_readable: "可从图中读数",
  display_format: "显示格式",
  missing_condition_note_zh: "缺条件说明",
  description: "给模型的说明",
  description_zh: "说明",
};
// Shown whatever their values; any other attribute gets a column only when some field sets it.
const LEADING = [
  "name", "label", "kind", "level", "entity", "canonical_unit", "categories", "cardinality", "range_policy",
  "references", "condition_rule", "valid_range",
];

// ---------- small builders: text only ----------

function el(tag, { className = "", text = null, title = null } = {}, ...children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = String(text);
  if (title) node.title = String(title);
  node.append(...children);
  return node;
}

function section(heading, ...children) {
  return el("section", { className: "profile-section" }, el("h2", { text: heading }), ...children);
}

const isEmpty = (value) => value == null || value === "" || (Array.isArray(value) && value.length === 0);

// One attribute value as plain text: a range as its ends, a list joined, a yes/no, anything else as JSON or itself.
function valueText(name, value) {
  if (isEmpty(value)) return "";
  if (name === "valid_range" && Array.isArray(value)) {
    const [low, high] = value;
    if (low == null && high == null) return "";
    return `${low ?? "−∞"} – ${high ?? "∞"}`;
  }
  if (typeof value === "boolean") return value ? "是" : "否";
  if (Array.isArray(value)) return value.map((item) => (typeof item === "object" ? JSON.stringify(item) : String(item))).join("、");
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

// A table from `columns` ({header, head, html, text}, the table.js shape) over `items`, with a copy button when asked.
function table(columns, items, { copy = false, className = "" } = {}) {
  const wrap = el("div", { className: "table-wrap" });
  const tableNode = el("table", { className: `facts-table profile-table ${className}`.trim() });
  const head = el("thead");
  const headRow = document.createElement("tr");
  headRow.innerHTML = columns.map((column) => column.head).join("");
  head.append(headRow);
  const body = el("tbody");
  for (const item of items) {
    const row = document.createElement("tr");
    row.innerHTML = columns.map((column) => column.html(item)).join("");
    body.append(row);
  }
  tableNode.append(head, body);
  wrap.append(tableNode);
  if (!copy) return wrap;
  const bar = el("div", { className: "results-head" }, copyButton(() => copyTable(columns, items)));
  return el("div", {}, bar, wrap);
}

// A text column: header over its attribute name, the cell's text escaped, `title` as the cell's tooltip.
function textColumn(header, text, { sub = "", mono = false, title = () => "" } = {}) {
  const small = sub ? `<small>${escapeHtml(sub)}</small>` : "";
  return {
    header,
    head: `<th>${escapeHtml(header)}${small}</th>`,
    html: (item) => {
      const shown = text(item);
      const tip = title(item);
      const classes = ["profile-cell", mono ? "mono" : "", shown ? "" : "empty"].filter(Boolean).join(" ");
      return `<td class="${classes}"${tip ? ` title="${escapeHtml(tip)}"` : ""}>${shown ? escapeHtml(shown) : "—"}</td>`;
    },
    text,
  };
}

// ---------- the definition ----------

// Draws `definition` into `root` (replacing what it held). `loadPrompts(field)` answers { sections: [{title, text}] }
// for the profile's system prompts (field null) or one field's question, or null when the answer is no longer
// wanted (the reader left); without it there is no prompt preview. `documentHref(id)` links a document that has
// results under the profile; without it that list is left out.
export function renderDefinition(root, definition, { loadPrompts = null, documentHref = null, documentName = (id) => id } = {}) {
  root.innerHTML = "";
  const ui = definition.ui ?? {};
  const entityLabel = new Map((definition.entities ?? []).map((entity) => [entity.name, entity.label_zh || entity.name]));
  const levelText = (level, entity) =>
    level === "paper" ? ui.paper_level_label_zh || "论文级" : `${entityLabel.get(entity) || ui.entity_label_zh || entity || ""}级`;

  root.append(header(definition));
  // A profile without entity types has one implicit, unlabelled entity, which the page names nowhere else either.
  const entities = definition.entities ?? [];
  if (entities.length > 1 || entities[0]?.label_zh) root.append(entitiesSection(definition));
  root.append(groupsSection(definition, levelText));
  root.append(fieldsSection(definition, levelText, entityLabel));
  root.append(unitsSection(definition.units ?? {}));
  root.append(retrievalSection(definition.retrieval ?? {}));
  if (documentHref && definition.finished_documents) root.append(documentsSection(definition.finished_documents, documentHref, documentName));
  if (loadPrompts) root.append(promptsSection(definition, loadPrompts));
}

function header(definition) {
  const title = el("h1", { text: definition.title_zh || definition.name });
  if (definition.maturity === "example") title.append(el("span", { className: "profile-badge", text: "示例配置" }));
  const meta = el(
    "p",
    { className: "profile-meta muted" },
    el("code", { text: definition.name ?? "" }),
    ` · 成熟度 ${definition.maturity ?? "—"} · 内容哈希 `,
    el("code", { text: definition.content_hash ?? "" }),
  );
  const box = el("header", { className: "profile-head" }, title, meta);
  if (definition.description_zh) box.append(el("p", { className: "profile-description", text: definition.description_zh }));
  if (definition.runnable === false) {
    box.append(el("p", { className: "profile-warning", text: `本服务器上不可运行：${definition.not_runnable ?? ""}` }));
  }
  if (definition.on_disk_changed) {
    box.append(el("p", { className: "profile-warning", text: "配置文件在服务器启动后被改动过；这里显示的是启动时读到的版本，重启后才会生效。" }));
  }
  return box;
}

function entitiesSection(definition) {
  const list = el("ul", { className: "profile-entities" });
  (definition.entities ?? []).forEach((entity, index) => {
    const item = el("li", {}, el("b", { text: entity.label_zh || entity.name }), " ", el("code", { text: entity.name }));
    const notes = [index === 0 ? "主实体" : "", `${(entity.fields ?? []).length} 个字段`];
    if (entity.sample_list_heading) notes.push(`样本清单标题「${entity.sample_list_heading}」`);
    item.append(el("span", { className: "muted", text: `（${notes.filter(Boolean).join("，")}）` }));
    list.append(item);
  });
  return section("实体类型", list);
}

function groupsSection(definition, levelText) {
  const columns = [
    textColumn("分组", (group) => group.name ?? "", { mono: true }),
    textColumn("显示名", (group) => group.label_zh ?? ""),
    textColumn("层级", (group) => levelText(group.level, group.entity)),
  ];
  return section("字段分组", table(columns, definition.groups ?? []));
}

function fieldsSection(definition, levelText, entityLabel) {
  const fields = definition.fields ?? [];
  const names = [...new Set(fields.flatMap((field) => Object.keys(field)))];
  const shown = [
    ...LEADING.filter((name) => names.includes(name)),
    ...names.filter((name) => !LEADING.includes(name) && fields.some((field) => !isEmpty(field[name]))),
  ];
  const cell = (name) => (field) => {
    if (name === "level") return levelText(field.level, field.entity);
    if ((name === "entity" || name === "references") && field[name]) return `${entityLabel.get(field[name]) ?? ""} ${field[name]}`.trim();
    return valueText(name, field[name]);
  };
  const columns = shown.map((name) =>
    textColumn(ATTRIBUTE_LABEL[name] ?? name, cell(name), {
      sub: name,
      mono: name === "name",
      title: name === "name" ? (field) => field.description_zh || field.description || "" : () => "",
    }),
  );
  const note = el("p", { className: "results-note muted", text: `${fields.length} 个字段。鼠标停在字段名上可看说明；空白（—）是未设置，取内置默认。` });
  return section("字段", note, table(columns, fields, { copy: true, className: "profile-fields" }));
}

function unitsSection(units) {
  const declared = units.declared ?? [];
  const aliasText = (alias) => {
    const factor = alias.factor != null && alias.factor !== 1 ? ` ×${alias.factor}` : "";
    const offset = alias.offset ? ` ${alias.offset > 0 ? "+" : "−"}${Math.abs(alias.offset)}` : "";
    return `${alias.spelling}${factor}${offset}`;
  };
  const columns = [
    textColumn("单位", (unit) => unit.canonical ?? "", { mono: true }),
    textColumn("写法（换算）", (unit) => (unit.aliases ?? []).map(aliasText).join("、")),
    textColumn("区分大小写", (unit) => valueText("", unit.case_sensitive)),
    textColumn("扩展内置", (unit) => valueText("", unit.extends_builtin)),
    textColumn("排除写法", (unit) => valueText("", unit.exclude)),
    textColumn("参与检索", (unit) => valueText("", unit.retrieval)),
  ];
  const body = declared.length ? table(columns, declared) : el("p", { className: "muted", text: "没有声明单位，只用内置的单位换算。" });
  const extra = el("dl", { className: "profile-list" });
  if ((units.ignored_suffixes ?? []).length) extra.append(el("dt", { text: "忽略的后缀" }), el("dd", { text: units.ignored_suffixes.join("、") }));
  if ((units.known ?? []).length) extra.append(el("dt", { text: "认得的单位" }), el("dd", { className: "mono", text: units.known.join("  ") }));
  return section("声明的单位", body, extra);
}

function retrievalSection(retrieval) {
  const list = el("dl", { className: "profile-list" });
  list.append(
    el("dt", { text: "条件关键词" }),
    el("dd", { text: valueText("", retrieval.condition_keywords) || "—" }),
    el("dt", { text: "条件单位模式" }),
    el("dd", { className: "mono", text: retrieval.condition_unit_pattern || "—" }),
  );
  return section("检索", list);
}

function documentsSection(ids, documentHref, documentName) {
  if (!ids.length) return section("已有结果的文档", el("p", { className: "muted", text: "还没有文档在这个领域下处理完成。" }));
  const list = el("ul", { className: "profile-documents" });
  for (const id of ids) {
    const link = el("a", { text: documentName(id) });
    link.href = documentHref(id);
    list.append(el("li", {}, link));
  }
  return section(`已有结果的文档（${ids.length}）`, list);
}

// ---------- prompt previews ----------

// Fetched only when opened: the system prompts once, a field's question each time another field is picked (the
// latest pick wins, whatever order the answers come back in).
function promptsSection(definition, loadPrompts) {
  const system = el("details", { className: "prompt-preview" }, el("summary", { text: "系统提示词" }));
  const systemBody = el("div", { className: "prompt-sections" });
  system.append(systemBody);
  let asked = false;
  system.addEventListener("toggle", () => {
    if (!system.open || asked) return;
    asked = true;
    fill(systemBody, () => loadPrompts(null), () => { asked = false; });
  });

  const fieldBox = el("details", { className: "prompt-preview" }, el("summary", { text: "单个字段的提问" }));
  const select = el("select", { className: "prompt-field" });
  select.setAttribute("aria-label", "选择字段");
  select.append(el("option", { text: "选择字段…" }));
  select.options[0].value = "";
  for (const field of definition.fields ?? []) {
    const option = el("option", { text: field.label ? `${field.label} ${field.name}` : field.name });
    option.value = field.name;
    select.append(option);
  }
  const fieldBody = el("div", { className: "prompt-sections" });
  let pick = 0;
  select.addEventListener("change", () => {
    const token = ++pick;
    const name = select.value;
    if (!name) {
      fieldBody.innerHTML = "";
      return;
    }
    fill(fieldBody, () => loadPrompts(name), () => {}, () => token === pick);
  });
  fieldBox.append(el("label", { className: "prompt-field-label" }, "字段 ", select), fieldBody);

  const note = el("p", {
    className: "results-note muted",
    text: "与命令行 paperfacts prompts 打印的完全相同；尖括号里的占位符在运行时由论文内容填入。",
  });
  return section("提示词预览", note, system, fieldBox);
}

async function fill(body, load, onError, wanted = () => true) {
  body.replaceChildren(el("p", { className: "muted", text: "正在读取…" }));
  let answer;
  try {
    answer = await load();
  } catch (error) {
    if (!wanted()) return;
    onError();
    body.replaceChildren(el("p", { className: "profile-warning", text: `读取提示词失败：${error.message}` }));
    return;
  }
  if (answer == null || !wanted()) return;
  body.replaceChildren(
    ...(answer.sections ?? []).map((part) => el("div", { className: "prompt-section" }, el("h3", { text: part.title }), el("pre", { text: part.text }))),
  );
}
