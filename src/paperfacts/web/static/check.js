// The profile check page (#/check): pasted profile JSON goes to POST /api/profile-check, and the answer is drawn as
// its errors, one per line, or, for a valid profile, its fields and the system prompts it would send. Nothing is
// stored; a local file is read in the browser and only its text is sent when 检查 is pressed.
//
// Everything drawn here is the pasted text's own (labels, descriptions, prompts, error lines quoting it), so it is
// untrusted: every string goes in through textContent, never innerHTML. The preview is profile.js's renderDefinition,
// so a valid profile is shown as literally the page it would get (which escapes the same way).

import { api } from "./api.js";
import { showPage } from "./document.js";
import { el } from "./html.js";
import { renderDefinition } from "./profile.js";

// The latest check: an older answer that lands after a newer press is dropped. Nothing else gates an answer: the page
// is profile-free and its text and result stay in the (hidden) page across navigation, so an answer that lands while
// the reader is elsewhere is drawn for the text it checked and is there when they come back.
let run = 0;

// What a failed check's status means for the reader: the server's detail says which, this says whether to retry.
const FAILURE_HINT = {
  408: "配置没有及时传完",
  413: "配置太大",
  429: "服务器正忙，请稍后再试",
  502: "检查进程出错，没有给出结果",
  503: "检查超时或暂时无法启动，可以稍后再试",
};

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
  const mine = ++run;
  const text = document.getElementById("check-text").value;
  const button = document.getElementById("check-run");
  if (!text.trim()) {
    setStatus("请先粘贴领域配置 JSON");
    return;
  }
  button.disabled = true;
  setStatus("检查中…");
  // The previous answer is about other text; it must not stand next to this one while it is checked.
  const root = document.getElementById("check-result");
  root.replaceChildren();
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
  if (failure) root.append(el("p", { className: "check-verdict bad", text: failureText(failure) }));
  else renderResult(root, result, { text, mine });
}

function failureText(failure) {
  const hint = FAILURE_HINT[failure.status];
  return `没能完成检查${hint ? `（${hint}）` : ""}：${failure.message}`;
}

function setStatus(text) {
  document.getElementById("check-status").textContent = text;
}

function renderResult(root, result, asked) {
  const errors = result.errors ?? [];
  root.append(
    el("p", {
      className: `check-verdict ${result.ok ? "ok" : "bad"}`,
      text: result.ok ? "✓ 配置有效，可以加载" : `✗ 发现 ${errors.length} 个问题，配置不能加载`,
    }),
  );
  if (errors.length) root.append(list("ol", "check-errors", errors));
  if (result.warnings?.length) root.append(el("h2", { text: "警告" }), list("ul", "check-notes", result.warnings));
  if (result.notes?.length) root.append(el("h2", { text: "说明" }), list("ul", "check-notes", result.notes));
  if (result.same_name) {
    const { name, same_content_hash: same } = result.same_name;
    root.append(
      el("p", {
        className: "check-same-name",
        text: same
          ? `与服务器上的「${name}」同名；内容哈希相同，只改了显示文字`
          : `与服务器上的「${name}」同名；内容哈希不同，改动会重新抽取`,
      }),
    );
  }
  if (result.definition) {
    const preview = el("div", { className: "check-preview" });
    root.append(el("h2", { text: "预览" }), preview);
    renderDefinition(preview, result.definition, { loadPrompts: (field) => loadPrompts(field, result, asked) });
  }
}

// The system prompts came with the check; a field's question is the same text checked again with ?field=. An answer
// that lands after another check is no longer wanted (null).
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
  return el(tag, { className }, ...items.map((item) => el("li", { text: item })));
}
