// The profile check page (#/check): pasted profile JSON goes to POST /api/profile-check, and the answer is drawn as
// its errors, one per line, or, for a valid profile, its fields and the system prompts it would send. Nothing is
// stored; a local file is read in the browser and only its text is sent when 检查 is pressed.
//
// Everything drawn here is the pasted text's own (labels, descriptions, prompts, error lines quoting it), so it is
// untrusted: every string goes in through textContent, never innerHTML.
//
// The preview is a minimal renderer of its own; once the profile page (profile.js) exports its definition and
// prompt renderers, this page should draw `definition` and `prompts` with those, so a preview is literally the page
// the profile would get.

import { api } from "./api.js";
import { showPage } from "./document.js";
import { isCurrent, state } from "./state.js";

let run = 0; // the latest check: an older answer that lands after a newer press is dropped

export function setupCheck() {
  const file = document.getElementById("check-file");
  document.getElementById("check-pick").addEventListener("click", () => file.click());
  file.addEventListener("change", () => {
    const chosen = file.files?.[0];
    file.value = "";
    if (!chosen) return;
    const reader = new FileReader();
    reader.addEventListener("load", () => {
      document.getElementById("check-text").value = String(reader.result ?? "");
      setStatus(`已读取 ${chosen.name}，按「检查」开始`);
    });
    reader.addEventListener("error", () => setStatus(`读取 ${chosen.name} 失败`));
    reader.readAsText(chosen);
  });
  document.getElementById("check-run").addEventListener("click", check);
}

// Router entry point. The pasted text stays in the textarea across navigation.
export function showCheck() {
  showPage("check-view");
}

async function check() {
  const generation = state.generation;
  const mine = ++run;
  const text = document.getElementById("check-text").value;
  const button = document.getElementById("check-run");
  if (!text.trim()) {
    setStatus("请先粘贴领域配置 JSON");
    return;
  }
  button.disabled = true;
  setStatus("检查中…");
  let result = null;
  let failure = null;
  try {
    result = await api("/api/profile-check", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: text,
    });
  } catch (error) {
    failure = error;
  }
  if (mine !== run) return;
  button.disabled = false;
  setStatus("");
  if (!isCurrent(generation)) return; // the reader left the page; the text and the button are still here for later
  const root = document.getElementById("check-result");
  root.replaceChildren();
  if (failure) root.append(element("p", "check-verdict bad", `没能完成检查：${failure.message}`));
  else renderResult(root, result);
}

function setStatus(text) {
  document.getElementById("check-status").textContent = text;
}

function renderResult(root, result) {
  const errors = result.errors ?? [];
  root.append(
    element(
      "p",
      `check-verdict ${result.ok ? "ok" : "bad"}`,
      result.ok ? "✓ 配置有效，可以加载" : `✗ 发现 ${errors.length} 个问题，配置不能加载`,
    ),
  );
  if (errors.length) root.append(list("ol", "check-errors", errors));
  if (result.warnings?.length) root.append(element("h2", "", "警告"), list("ul", "check-notes", result.warnings));
  if (result.notes?.length) root.append(element("h2", "", "说明"), list("ul", "check-notes", result.notes));
  if (result.same_name) {
    const { name, same_content_hash: same } = result.same_name;
    root.append(
      element(
        "p",
        "check-same-name",
        same
          ? `与服务器上的「${name}」同名；内容哈希相同，只改了显示文字`
          : `与服务器上的「${name}」同名；内容哈希不同，改动会重新抽取`,
      ),
    );
  }
  if (result.definition) renderDefinition(root, result.definition);
  if (result.prompts?.length) renderPrompts(root, result.prompts);
}

// The profile's heading and its field table.
function renderDefinition(root, definition) {
  root.append(element("h2", "", "预览"));
  const head = element("p", "check-profile-head");
  head.append(
    element("b", "", definition.title_zh || definition.name),
    document.createTextNode(` · ${definition.name} · 内容哈希 `),
    element("code", "", definition.content_hash ?? ""),
    document.createTextNode(definition.maturity === "example" ? " · 示例配置" : ""),
  );
  root.append(head);
  if (definition.description_zh) root.append(element("p", "muted", definition.description_zh));
  if (definition.entities?.length > 1) {
    root.append(list("ul", "check-notes", definition.entities.map((entity) => `${entity.label_zh || entity.name}（${entity.name}）：${entity.fields.length} 个字段`)));
  }
  const columns = [
    ["字段", (field) => field.name],
    ["名称", (field) => field.label],
    ["分组", (field) => field.group],
    ["层级", (field) => field.level],
    ["类型", (field) => field.kind],
    ["单位", (field) => field.canonical_unit ?? ""],
    ["合理范围", (field) => range(field.valid_range)],
    ["说明", (field) => field.description],
  ];
  const table = element("table", "facts-table check-fields");
  const headRow = element("tr");
  for (const [title] of columns) headRow.append(element("th", "", title));
  table.append(element("thead"));
  table.tHead.append(headRow);
  const body = element("tbody");
  for (const field of definition.fields ?? []) {
    const row = element("tr");
    for (const [, value] of columns) row.append(element("td", "", String(value(field) ?? "")));
    body.append(row);
  }
  table.append(body);
  const wrap = element("div", "table-wrap");
  wrap.append(table);
  root.append(wrap);
}

function range(bounds) {
  if (!bounds || (bounds[0] == null && bounds[1] == null)) return "";
  return `${bounds[0] ?? "−∞"} – ${bounds[1] ?? "∞"}`;
}

// Each system prompt, exactly as the model would get it.
function renderPrompts(root, prompts) {
  root.append(element("h2", "", "系统提示词"));
  for (const section of prompts) {
    const details = element("details", "check-prompt");
    details.append(element("summary", "", section.title), element("pre", "", section.text));
    root.append(details);
  }
}

function list(tag, className, items) {
  const root = element(tag, className);
  for (const item of items) root.append(element("li", "", String(item)));
  return root;
}

function element(tag, className = "", text = null) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text != null) node.textContent = text;
  return node;
}
