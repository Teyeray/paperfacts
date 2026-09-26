// The profile check page (#/check): pasted profile JSON goes to POST /api/profile-check, and the answer is drawn as
// its errors, one per line, or, for a valid profile, its fields and the system prompts it would send. Nothing is
// stored; a local file is read in the browser and only its text is sent when 检查 is pressed.
//
// Everything drawn here is the pasted text's own (labels, descriptions, prompts, error lines quoting it), so it is
// untrusted: every string goes in through textContent, never innerHTML. The preview is profile.js's renderDefinition,
// so a valid profile is shown as literally the page it would get (which escapes the same way).

import { api } from "./api.js";
import { showPage } from "./document.js";
import { renderDefinition } from "./profile.js";
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
  else renderResult(root, result, { text, mine });
}

function setStatus(text) {
  document.getElementById("check-status").textContent = text;
}

function renderResult(root, result, asked) {
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
  if (result.definition) {
    const preview = element("div", "check-preview");
    root.append(element("h2", "", "预览"), preview);
    renderDefinition(preview, result.definition, { loadPrompts: (field) => loadPrompts(field, result, asked) });
  }
}

// The system prompts came with the check; a field's question is the same text checked again with ?field=. An answer
// that lands after another check is no longer wanted (null); leaving the page does not matter, since the preview
// stays in the (hidden) page with the pasted text.
async function loadPrompts(field, result, { text, mine }) {
  if (field == null) return { sections: result.prompts ?? [] };
  const answer = await api(`/api/profile-check?field=${encodeURIComponent(field)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: text,
  });
  if (mine !== run) return null;
  return { sections: answer.prompts ?? [] };
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
