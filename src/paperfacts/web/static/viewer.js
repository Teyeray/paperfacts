// Page provenance viewer: the page image plus an SVG overlay. Each block is one <rect>, whose
// coordinates come from the normalized bbox × the image's pixel dimensions — the same conversion
// as the backend's NormalizedBBox.to_pixels, so the boxes drawn here line up with the overlay and
// cropping tools.

import { escapeHtml } from "./html.js";
import { LANES, LANE_LABEL } from "./state.js";

const LANE_CLASS = { mineru: "a", paddleocr_vl: "b" };

export class PageViewer {
  // initial: the previous instance's getState(), used so a re-render of the document view keeps
  // showing the same page and the same highlight set
  constructor(root, { documentId, pageCount, blocks, dpi = 110, pdfAvailable = true, initial = null }) {
    this.root = root;
    this.documentId = documentId;
    this.pageCount = pageCount;
    this.blocks = blocks; // { mineru: [...], paddleocr_vl: [...] }
    this.dpi = dpi;
    this.pdfAvailable = pdfAvailable;
    this.page = Math.min(initial?.page ?? 0, pageCount - 1);
    this.highlighted = new Set(initial?.highlighted ?? []);
    this.show = { ...Object.fromEntries(LANES.map((l) => [l, true])), ...(initial?.show ?? {}) };
    this.dim = initial?.dim ?? true;
    this._render();
  }

  // ---- public ----
  getState() {
    return { page: this.page, highlighted: [...this.highlighted], show: { ...this.show }, dim: this.dim };
  }

  highlight(sourceIds, { jump = true } = {}) {
    this.highlighted = new Set(sourceIds);
    const pages = this.highlightedPages();
    if (jump && pages.length && !pages.includes(this.page)) this.page = pages[0];
    this._render();
  }

  showPage(page) {
    if (page < 0 || page >= this.pageCount) return;
    this.page = page;
    this._render();
  }

  highlightedPages() {
    const pages = new Set();
    for (const lane of Object.keys(this.blocks)) {
      for (const block of this.blocks[lane]) if (this.highlighted.has(block.source_id)) pages.add(block.page);
    }
    return [...pages].sort((a, b) => a - b);
  }

  // ---- rendering ----
  _render() {
    this.root.innerHTML = "";
    this.root.append(this._bar());
    if (!this.pdfAvailable) {
      const empty = document.createElement("div");
      empty.className = "viewer-empty";
      empty.textContent = "这篇文档不是网页上传的，且原 PDF 路径在本机不存在：无法渲染页面，但表格里的 source_id 仍然有效。";
      this.root.append(empty);
      return;
    }
    const page = document.createElement("div");
    page.className = "page" + (this.dim && this.highlighted.size ? " dim" : "");
    const img = document.createElement("img");
    img.alt = `第 ${this.page + 1} 页`;
    img.src = `/api/documents/${this.documentId}/pages/${this.page}.png?dpi=${this.dpi}`;
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.setAttribute("preserveAspectRatio", "none");
    img.addEventListener("load", () => {
      svg.setAttribute("viewBox", `0 0 ${img.naturalWidth} ${img.naturalHeight}`);
      this._drawBlocks(svg, img.naturalWidth, img.naturalHeight, page);
    });
    page.append(img, svg);
    this.root.append(page);
  }

  _bar() {
    const bar = document.createElement("div");
    bar.className = "viewer-bar";
    const pager = document.createElement("span");
    pager.className = "pager";
    const prev = button("‹", () => this.showPage(this.page - 1), this.page === 0);
    const next = button("›", () => this.showPage(this.page + 1), this.page >= this.pageCount - 1);
    const label = document.createElement("span");
    label.textContent = `${this.page + 1} / ${this.pageCount}`;
    pager.append(prev, label, next);
    bar.append(pager);

    for (const lane of LANES) {
      const l = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = this.show[lane];
      cb.addEventListener("change", () => { this.show[lane] = cb.checked; this._render(); });
      const sw = document.createElement("span");
      sw.className = `swatch ${LANE_CLASS[lane]}`;
      l.append(cb, sw, document.createTextNode(` ${LANE_LABEL[lane]} ${this.blocks[lane]?.length ?? 0} 块`));
      bar.append(l);
    }
    const dimLabel = document.createElement("label");
    const dimCb = document.createElement("input");
    dimCb.type = "checkbox";
    dimCb.checked = this.dim;
    dimCb.addEventListener("change", () => { this.dim = dimCb.checked; this._render(); });
    dimLabel.append(dimCb, document.createTextNode(" 只突出选中"));
    bar.append(dimLabel);

    const pages = this.highlightedPages();
    if (pages.length) {
      const hl = document.createElement("span");
      hl.className = "hl-pages";
      hl.append(document.createTextNode("来源页："));
      for (const p of pages) hl.append(button(`p${p + 1}`, () => this.showPage(p), false));
      bar.append(hl);
    }
    return bar;
  }

  _drawBlocks(svg, width, height, container) {
    svg.innerHTML = "";
    let tooltip = null;
    for (const lane of LANES) {
      if (!this.show[lane]) continue;
      for (const block of this.blocks[lane] ?? []) {
        if (block.page !== this.page) continue;
        const { x1, y1, x2, y2 } = block.bbox;
        const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
        rect.setAttribute("x", (x1 * width).toFixed(1));
        rect.setAttribute("y", (y1 * height).toFixed(1));
        rect.setAttribute("width", ((x2 - x1) * width).toFixed(1));
        rect.setAttribute("height", ((y2 - y1) * height).toFixed(1));
        rect.setAttribute("vector-effect", "non-scaling-stroke");
        rect.setAttribute("class", `${LANE_CLASS[lane]}${this.highlighted.has(block.source_id) ? " hl" : ""}`);
        rect.addEventListener("mouseenter", (event) => {
          tooltip = document.createElement("div");
          tooltip.className = "tooltip";
          tooltip.innerHTML = `<code>${escapeHtml(block.source_id)}</code> · ${escapeHtml(block.type)}<br>${escapeHtml(block.content.slice(0, 160))}`;
          container.append(tooltip);
          positionTooltip(tooltip, event, container);
        });
        rect.addEventListener("mousemove", (event) => tooltip && positionTooltip(tooltip, event, container));
        rect.addEventListener("mouseleave", () => { tooltip?.remove(); tooltip = null; });
        svg.append(rect);
      }
    }
    // draw the highlighted ones on top
    for (const hl of [...svg.querySelectorAll("rect.hl")]) svg.append(hl);
  }
}

function button(text, onClick, disabled) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = text;
  b.disabled = disabled;
  b.addEventListener("click", onClick);
  return b;
}

function positionTooltip(tooltip, event, container) {
  const box = container.getBoundingClientRect();
  const x = event.clientX - box.left + 12;
  const y = event.clientY - box.top + 12;
  tooltip.style.left = `${Math.min(x, box.width - 330)}px`;
  tooltip.style.top = `${y}px`;
}
