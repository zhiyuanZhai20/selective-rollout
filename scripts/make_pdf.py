"""
Render results/report_zh.md -> results/report_zh.pdf via matplotlib PdfPages.

Layout model: each page has a single full-page axes with x in [0, PAGE_W],
y in [0, PAGE_H] (inches, origin bottom-left). We track a top-origin cursor
y_in, and convert to axes y via (PAGE_H - y_in).
All text + rectangles go onto that axes so draw order is predictable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Rectangle
from PIL import Image

plt.rcParams["font.family"] = ["DejaVu Sans", "Droid Sans Fallback"]
plt.rcParams["font.size"] = 10
plt.rcParams["pdf.fonttype"] = 42

ROOT = Path(__file__).resolve().parent.parent
MD_PATH = ROOT / "results" / "report_zh.md"
OUT_PATH = ROOT / "results" / "report_zh.pdf"
FIG_DIR = ROOT / "results" / "figures"

PAGE_W, PAGE_H = 8.27, 11.69
MARGIN_L, MARGIN_R = 0.85, 0.85
MARGIN_T, MARGIN_B = 0.85, 0.85
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R
CONTENT_H = PAGE_H - MARGIN_T - MARGIN_B


@dataclass
class Block:
    kind: str
    text: str = ""
    depth: int = 0
    lines: list = field(default_factory=list)
    header: list = field(default_factory=list)
    rows: list = field(default_factory=list)


def _strip_md(s: str) -> str:
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"\1", s)
    s = s.replace("$", r"\$")
    return s


def parse_md(md_text: str) -> list[Block]:
    lines = md_text.splitlines()
    i = 0
    blocks: list[Block] = []
    while i < len(lines):
        ln = lines[i]
        s = ln.strip()

        if s.startswith("# "):
            blocks.append(Block("h1", text=s[2:].strip())); i += 1; continue
        if s.startswith("## "):
            blocks.append(Block("h2", text=s[3:].strip())); i += 1; continue
        if s.startswith("### "):
            blocks.append(Block("h3", text=s[4:].strip())); i += 1; continue

        if s.startswith("```"):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            i += 1
            blocks.append(Block("code", lines=code))
            continue

        if "|" in ln and i + 1 < len(lines) and re.match(r"^\s*\|?[\s\-:|]+\|?\s*$", lines[i + 1]):
            header = [c.strip() for c in ln.strip().strip("|").split("|")]
            i += 2
            body = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                row = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                if len(row) != len(header):
                    break
                body.append(row); i += 1
            blocks.append(Block("table", header=header, rows=body))
            continue

        m = re.match(r"^(\s*)[-*]\s+(.*)$", ln)
        if m:
            blocks.append(Block("bullet", text=m.group(2), depth=len(m.group(1)) // 2))
            i += 1; continue

        m = re.match(r"^(\s*)\d+\.\s+(.*)$", ln)
        if m:
            blocks.append(Block("bullet", text=m.group(2), depth=len(m.group(1)) // 2))
            i += 1; continue

        if re.match(r"^-{3,}\s*$", s):
            blocks.append(Block("hrule")); i += 1; continue

        if not s:
            i += 1; continue

        para = [ln]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if (not nxt.strip()
                or nxt.lstrip().startswith(("#", "-", "*", "```", ">"))
                or re.match(r"^\s*\d+\.\s+", nxt)
                or "|" in nxt):
                break
            para.append(nxt); j += 1
        blocks.append(Block("para", text=" ".join(p.strip() for p in para)))
        i = j

    return blocks


def _char_w_in(ch: str, fontsize_pt: float) -> float:
    """Approximate character width in inches at the given font size.
    CJK/CJK-punct glyphs are full-em; Latin is ~0.55em in DejaVu Sans."""
    em_in = fontsize_pt / 72.0
    cp = ord(ch)
    if cp > 0x2E80 or 0xFF00 <= cp <= 0xFFEF:  # CJK + fullwidth forms
        return em_in * 1.0
    # rough per-char widths in em for DejaVu Sans at 10pt
    narrow = set(" .,;:!|'`i l")
    if ch in narrow:
        return em_in * 0.30
    if ch.isdigit():
        return em_in * 0.55
    if ch.isupper():
        return em_in * 0.70
    return em_in * 0.55


def _wrap(text: str, max_w_in: float, fontsize_pt: float = 10.5) -> list[str]:
    """Greedy wrap so each output line fits max_w_in at fontsize_pt."""
    out, cur, cur_w = [], [], 0.0
    for ch in text:
        w = _char_w_in(ch, fontsize_pt)
        if cur_w + w > max_w_in and cur:
            out.append("".join(cur)); cur = [ch]; cur_w = w
        else:
            cur.append(ch); cur_w += w
    if cur:
        out.append("".join(cur))
    return out or [""]


def _fit_single_line(text: str, max_w_in: float, fontsize_pt: float) -> str:
    """Truncate with … if too wide for one line."""
    total = sum(_char_w_in(c, fontsize_pt) for c in text)
    if total <= max_w_in:
        return text
    ell_w = _char_w_in("…", fontsize_pt)
    budget = max_w_in - ell_w
    out = []
    used = 0.0
    for ch in text:
        cw = _char_w_in(ch, fontsize_pt)
        if used + cw > budget:
            break
        out.append(ch); used += cw
    return "".join(out) + "…"


class Renderer:
    def __init__(self):
        self.pdf = PdfPages(OUT_PATH)
        self.fig = None
        self.ax = None
        self.y_in = 0.0
        self.page_num = 0
        self._new_page()

    # --- coord helpers (inches, origin top-left of content area) ---
    def _xy(self, x_in, y_in_from_top):
        # to axes coords (inches, origin bottom-left of page)
        return MARGIN_L + x_in, PAGE_H - MARGIN_T - y_in_from_top

    def _new_page(self):
        if self.fig is not None:
            self._draw_footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
        self.fig = plt.figure(figsize=(PAGE_W, PAGE_H), dpi=150)
        self.fig.patch.set_facecolor("white")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, PAGE_W)
        self.ax.set_ylim(0, PAGE_H)
        self.ax.set_axis_off()
        self.y_in = 0.0
        self.page_num += 1

    def _draw_footer(self):
        self.ax.text(PAGE_W / 2, MARGIN_B / 2, f"- {self.page_num} -",
                     ha="center", va="center", color="#888", fontsize=8)

    def _ensure(self, need_in: float):
        if self.y_in + need_in > CONTENT_H:
            self._new_page()

    def _text(self, x_in, y_top_in, text, **kw):
        x, y = self._xy(x_in, y_top_in)
        self.ax.text(x, y, text, **kw)

    def _rect(self, x_in, y_top_in, w_in, h_in, facecolor=None,
              edgecolor=None, linewidth=0.5):
        x, y_top = self._xy(x_in, y_top_in)
        # y_top is the top; rectangle origin is bottom-left
        y_bot = y_top - h_in
        r = Rectangle((x, y_bot), w_in, h_in,
                      facecolor=facecolor if facecolor else "none",
                      edgecolor=edgecolor if edgecolor else "none",
                      linewidth=linewidth)
        self.ax.add_patch(r)

    def _hline(self, x_in, y_top_in, w_in, color="#bbb", lw=0.5):
        x, y = self._xy(x_in, y_top_in)
        self.ax.plot([x, x + w_in], [y, y], color=color, linewidth=lw)

    def _vline(self, x_in, y_top_in, h_in, color="#bbb", lw=0.5):
        x, y_top = self._xy(x_in, y_top_in)
        self.ax.plot([x, x], [y_top, y_top - h_in], color=color, linewidth=lw)

    # --- blocks ---
    def h1(self, text):
        self._ensure(0.7)
        self.y_in += 0.1
        self._text(0, self.y_in + 0.28, _strip_md(text),
                   fontsize=20, fontweight="bold", color="#111",
                   va="baseline")
        self.y_in += 0.45
        self._hline(0, self.y_in, CONTENT_W, color="#333", lw=0.8)
        self.y_in += 0.15

    def h2(self, text):
        self._ensure(0.6)
        self.y_in += 0.18
        self._text(0, self.y_in + 0.22, _strip_md(text),
                   fontsize=14, fontweight="bold", color="#143c78",
                   va="baseline")
        self.y_in += 0.4

    def h3(self, text):
        self._ensure(0.45)
        self.y_in += 0.1
        self._text(0, self.y_in + 0.18, _strip_md(text),
                   fontsize=11.5, fontweight="bold", color="#333",
                   va="baseline")
        self.y_in += 0.3

    def para(self, text):
        text = _strip_md(text)
        for line in _wrap(text, CONTENT_W, 10.5):
            self._ensure(0.22)
            self._text(0, self.y_in + 0.16, line,
                       fontsize=10.5, color="#111", va="baseline")
            self.y_in += 0.22
        self.y_in += 0.05

    def bullet(self, text, depth=0):
        text = _strip_md(text)
        indent = 0.18 + depth * 0.28
        tx = indent + 0.18
        lines = _wrap(text, CONTENT_W - tx, 10.5)
        for i, line in enumerate(lines):
            self._ensure(0.22)
            if i == 0:
                self._text(indent, self.y_in + 0.16, "•",
                           fontsize=10.5, color="#111", va="baseline")
            self._text(tx, self.y_in + 0.16, line,
                       fontsize=10.5, color="#111", va="baseline")
            self.y_in += 0.22
        self.y_in += 0.02

    def hrule(self):
        self._ensure(0.15)
        self.y_in += 0.05
        self._hline(0, self.y_in, CONTENT_W, color="#ccc", lw=0.6)
        self.y_in += 0.1

    def code(self, lines):
        line_h = 0.17
        max_half = CONTENT_W / 0.058  # 8.5pt mono-ish
        # expand lines that are too long via truncation (code text reads fine if trimmed with …)
        display_lines = []
        for ln in lines:
            s = ln.rstrip()
            # don't wrap, just truncate if overly long
            if len(s) * 0.058 > CONTENT_W - 0.2:
                limit = int((CONTENT_W - 0.25) / 0.058)
                s = s[:limit] + "…"
            display_lines.append(s)

        # paginate
        idx = 0
        while idx < len(display_lines):
            self._ensure(line_h + 0.2)
            seg_top = self.y_in
            seg_lines = []
            while idx < len(display_lines) and self.y_in + line_h + 0.1 <= CONTENT_H:
                seg_lines.append(display_lines[idx])
                self.y_in += line_h
                idx += 1
            # draw background for this segment (before text, but matplotlib
            # draws in insertion order — we already advanced y_in; draw rect now
            # then text; z-order defaults put rect behind text in our axes)
            pad_top = 0.04
            pad_bot = 0.04
            self._rect(0, seg_top, CONTENT_W,
                       self.y_in - seg_top + pad_bot,
                       facecolor="#f2f2f2", edgecolor="#e2e2e2", linewidth=0.3)
            # draw text on top (re-walk the segment)
            y_walk = seg_top + pad_top
            for s in seg_lines:
                self._text(0.12, y_walk + 0.13, s,
                           fontsize=8.5, color="#222", va="baseline")
                y_walk += line_h
            self.y_in += pad_bot
            if idx < len(display_lines):
                self._new_page()
        self.y_in += 0.08

    def table(self, header, rows):
        n = len(header)
        col_w = CONTENT_W / n
        cell_pad = 0.06
        inner_w = col_w - 2 * cell_pad
        hdr_fs = 8.5
        body_fs = 8.5
        # wrap headers
        header_lines_per_col = [
            _wrap(_strip_md(h), inner_w, hdr_fs) for h in header
        ]
        header_h = max(0.3, 0.16 + 0.14 * max(len(l) for l in header_lines_per_col))
        # wrap body rows
        body_lines = []
        row_heights = []
        for row in rows:
            cells = []
            max_lines = 1
            for v in row:
                wrapped = _wrap(_strip_md(v), inner_w, body_fs)
                cells.append(wrapped)
                max_lines = max(max_lines, len(wrapped))
            body_lines.append(cells)
            row_heights.append(0.12 + 0.14 * max_lines)

        i = 0
        while i < len(rows):
            # need at least header + one row
            self._ensure(header_h + row_heights[i] + 0.05)
            start_y = self.y_in
            # draw header fill + text
            self._rect(0, start_y, CONTENT_W, header_h,
                       facecolor="#3c5a8c", edgecolor="#2b426a", linewidth=0.4)
            for j, hlines in enumerate(header_lines_per_col):
                cx = j * col_w + col_w / 2
                total_h = 0.14 * (len(hlines) - 1)
                y0 = start_y + header_h / 2 - total_h / 2 + 0.05
                for k, line in enumerate(hlines):
                    self._text(cx, y0 + k * 0.14, line,
                               ha="center", va="baseline",
                               fontsize=hdr_fs, fontweight="bold", color="white")
            self.y_in += header_h

            first_in_page = i
            # draw rows that fit
            while i < len(rows) and self.y_in + row_heights[i] <= CONTENT_H:
                row_top = self.y_in
                rh = row_heights[i]
                # banding
                if (i - first_in_page) % 2 == 0:
                    self._rect(0, row_top, CONTENT_W, rh,
                               facecolor="#f5f5fa", edgecolor="none")
                cells = body_lines[i]
                for j, wrapped in enumerate(cells):
                    x0 = j * col_w + cell_pad
                    for k, line in enumerate(wrapped):
                        self._text(x0, row_top + 0.15 + k * 0.14, line,
                                   fontsize=body_fs, color="#111", va="baseline")
                self.y_in += rh
                i += 1

            # draw border around the whole table section on this page
            end_y = self.y_in
            self._rect(0, start_y, CONTENT_W, end_y - start_y,
                       facecolor="none", edgecolor="#aaa", linewidth=0.5)
            # vertical dividers
            for j in range(1, n):
                self._vline(j * col_w, start_y, end_y - start_y, color="#ccc", lw=0.3)
            # horizontal divider below header
            self._hline(0, start_y + header_h, CONTENT_W, color="#aaa", lw=0.4)

            self.y_in += 0.1
            if i < len(rows):
                self._new_page()

    def image_block(self, path: Path, caption=None):
        if not Path(path).exists():
            return
        img = Image.open(path)
        iw, ih = img.size
        w_in = CONTENT_W * 0.92
        h_in = w_in * ih / iw
        max_h = CONTENT_H * 0.72
        if h_in > max_h:
            h_in = max_h; w_in = h_in * iw / ih
        needed = h_in + (0.3 if caption else 0.1)
        self._ensure(needed)
        x_in = (CONTENT_W - w_in) / 2
        # place image via its own inset axes
        x, y_top = self._xy(x_in, self.y_in)
        y_bot = y_top - h_in
        ax_img = self.fig.add_axes(
            [x / PAGE_W, y_bot / PAGE_H, w_in / PAGE_W, h_in / PAGE_H],
            frameon=False,
        )
        ax_img.imshow(img)
        ax_img.set_axis_off()
        self.y_in += h_in + 0.05
        if caption:
            self._text(CONTENT_W / 2, self.y_in + 0.15, caption,
                       ha="center", fontsize=9, color="#666",
                       style="italic", va="baseline")
            self.y_in += 0.28
        self.y_in += 0.1

    def close(self):
        if self.fig is not None:
            self._draw_footer()
            self.pdf.savefig(self.fig)
            plt.close(self.fig)
            self.fig = None
        self.pdf.close()


def main():
    md = MD_PATH.read_text()
    blocks = parse_md(md)
    r = Renderer()

    for b in blocks:
        if b.kind == "h1": r.h1(b.text)
        elif b.kind == "h2": r.h2(b.text)
        elif b.kind == "h3": r.h3(b.text)
        elif b.kind == "para": r.para(b.text)
        elif b.kind == "bullet": r.bullet(b.text, b.depth)
        elif b.kind == "code": r.code(b.lines)
        elif b.kind == "hrule": r.hrule()
        elif b.kind == "table": r.table(b.header, b.rows)

    r._new_page()
    r.h2("附录：关键图")
    for caption, f in [
        ("ρ / AUROC 双指标热力图（所有 metric × K 组合）",
         FIG_DIR / "heatmap_rho_auroc.png"),
        ("三分层散点（prefix_edit_distance_mean @ K=15）",
         FIG_DIR / "stratified_prefix_edit_distance_mean_K15.png"),
        ("零方差 vs 非零方差：各 metric × K 的分布对比",
         FIG_DIR / "hist_zero_vs_nonzero.png"),
        ("scatter @ K=10", FIG_DIR / "scatter_K10.png"),
        ("scatter @ K=15", FIG_DIR / "scatter_K15.png"),
    ]:
        r.image_block(f, caption=caption)

    r.close()
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
