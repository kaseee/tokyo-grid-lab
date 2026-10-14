"""共通ウィジェット：カード、指標タイル、グラフ領域（ホバー付き）、スライダー など"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ui import theme
from ui.theme import C, FONTS


class Card(tk.Frame):
    """白背景＋細い枠線のカード。title を与えると見出し行を付ける。"""

    def __init__(self, parent, title: str | None = None, subtitle: str | None = None, pad=14, **kw):
        super().__init__(parent, bg=C["card"], highlightthickness=1, highlightbackground=C["border"],
                         highlightcolor=C["border"], **kw)
        self.pad = pad
        self.header = None
        if title:
            self.header = tk.Frame(self, bg=C["card"])
            self.header.pack(fill="x", padx=pad, pady=(pad - 2, 0))
            tk.Label(self.header, text=title, font=FONTS["h3"], bg=C["card"], fg=C["ink"]).pack(side="left")
            if subtitle:
                tk.Label(self.header, text=subtitle, font=FONTS["small"], bg=C["card"], fg=C["muted"]).pack(
                    side="left", padx=(10, 0), pady=(2, 0))
        self.body = tk.Frame(self, bg=C["card"])
        self.body.pack(fill="both", expand=True, padx=pad, pady=(8 if title else pad, pad))


class StatTile(tk.Frame):
    """指標タイル：ラベル／値／補足"""

    def __init__(self, parent, label: str, value: str = "-", sub: str = "", accent: str | None = None, small=False, **kw):
        super().__init__(parent, bg=C["card"], highlightthickness=1, highlightbackground=C["border"], **kw)
        inner = tk.Frame(self, bg=C["card"])
        inner.pack(fill="both", expand=True, padx=14, pady=(10, 11))
        top = tk.Frame(inner, bg=C["card"])
        top.pack(fill="x")
        if accent:
            tk.Frame(top, bg=accent, width=4, height=12).pack(side="left", padx=(0, 6), pady=(2, 0))
        self._label = tk.Label(top, text=label, font=FONTS["small"], fg=C["ink2"], bg=C["card"], anchor="w")
        self._label.pack(side="left", fill="x")
        self._value = tk.Label(inner, text=value, font=FONTS["stat_s" if small else "stat"], fg=C["ink"], bg=C["card"], anchor="w")
        self._value.pack(fill="x", pady=(2, 0))
        self._sub = tk.Label(inner, text=sub, font=FONTS["small"], fg=C["muted"], bg=C["card"], anchor="w",
                             justify="left")
        self._sub.pack(fill="x")
        self.bind("<Configure>", lambda e: self._sub.configure(wraplength=max(e.width - 30, 80)))

    def set(self, value: str | None = None, sub: str | None = None, sub_color: str | None = None,
            label: str | None = None) -> None:
        if value is not None:
            self._value.configure(text=value)
        if sub is not None:
            self._sub.configure(text=sub, fg=sub_color or C["muted"])
        if label is not None:
            self._label.configure(text=label)


class ChartCanvas(tk.Frame):
    """matplotlib Figure を埋め込む。ホバー（十字線＋ツールチップ）を簡単に付けられる。"""

    def __init__(self, parent, height=3.2, width=8.0, dpi=None, **kw):
        super().__init__(parent, bg=C["card"], **kw)
        dpi = dpi or 100  # 高 DPI 画面では matplotlib(TkAgg) が自動で拡大する
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor=C["card"], layout="constrained")
        self.fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, hspace=0.04, wspace=0.06)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        w = self.canvas.get_tk_widget()
        w.configure(bg=C["card"], highlightthickness=0, bd=0)
        w.pack(fill="both", expand=True)
        self._hovers = []
        self._tip = None
        self._vline = None
        self._last = None
        w.bind("<Map>", lambda e: self.after_idle(self._sync_size), add="+")
        self.fig._draw_hooks = []
        self.canvas.mpl_connect("draw_event", self._on_draw)
        self.canvas.mpl_connect("motion_notify_event", self._on_move)
        self.canvas.mpl_connect("figure_leave_event", self._on_leave)

    def _on_draw(self, _e):
        changed = False
        for hook in list(getattr(self.fig, "_draw_hooks", [])):
            try:
                changed = bool(hook()) or changed
            except Exception:
                pass
        if changed:
            self.canvas.draw_idle()

    def clear(self):
        self.fig.clear()
        self.fig._draw_hooks = []
        self._hovers = []
        self._tip = None
        self._vline = None
        self._last = None

    def _sync_size(self, redraw=True):
        """Figure の大きさをウィジェットの実寸に合わせる（非表示タブで描いた場合のずれ対策）。
        FigureCanvasTkAgg.resize を使い、内部の PhotoImage も同時に作り直す。"""
        w = self.canvas.get_tk_widget()
        W, H = w.winfo_width(), w.winfo_height()
        if W > 10 and H > 10:
            cur = self.fig.get_size_inches() * self.fig.dpi
            photo = getattr(self.canvas, "_tkphoto", None)
            pw = int(photo.width()) if photo is not None else W
            ph = int(photo.height()) if photo is not None else H
            if abs(cur[0] - W) > 0.5 or abs(cur[1] - H) > 0.5 or pw != W or ph != H:
                from types import SimpleNamespace
                self.canvas.resize(SimpleNamespace(width=W, height=H))
                return
        if redraw:
            self.canvas.draw_idle()

    def draw(self, layout=True):
        self._sync_size(redraw=True)

    # ---- ホバー
    def add_hover(self, ax, xs, rows_fn, title_fn=None, vline=True, span_axes=None, axis="x"):
        """
        xs: x 座標配列（matplotlib の数値座標）
        rows_fn(i) -> [(label, value_text, color or None), ...]
        title_fn(i) -> 見出し文字列
        span_axes: 十字線を引く軸のリスト（省略時は ax のみ）
        """
        self._hovers.append(dict(ax=ax, xs=np.asarray(xs, dtype=float), rows=rows_fn, title=title_fn,
                                 vline=vline and axis == "x", span=span_axes or [ax], axis=axis))

    def _on_leave(self, _e=None):
        if self._tip is not None:
            self._tip.set_visible(False)
        for ln in self._vline or []:
            ln.set_visible(False)
        self._last = None
        self.canvas.draw_idle()

    def _on_move(self, e):
        if e.inaxes is None or e.xdata is None:
            if self._last is not None:
                self._on_leave()
            return
        h = next((h for h in self._hovers if e.inaxes in h["span"]), None)
        if h is None:
            if self._last is not None:
                self._on_leave()
            return
        xs = h["xs"]
        pos = e.ydata if h["axis"] == "y" else e.xdata
        i = int(np.clip(np.searchsorted(xs, pos), 0, len(xs) - 1))
        if i > 0 and abs(xs[i - 1] - pos) < abs(xs[i] - pos):
            i -= 1
        if h["axis"] == "y" and abs(xs[i] - pos) > 0.5:
            if self._last is not None:
                self._on_leave()
            return
        key = (id(h), i)
        if key == self._last:
            return
        self._last = key
        rows = h["rows"](i)
        title = h["title"](i) if h["title"] else ""
        if self._vline:
            for ln in self._vline:
                try:
                    ln.remove()
                except Exception:
                    pass
        self._vline = []
        if h["vline"]:
            for a in h["span"]:
                ln = a.axvline(xs[i], color=C["muted"], lw=0.8, zorder=10)
                ln.set_in_layout(False)
                self._vline.append(ln)
        text = "\n".join([title] + [f"{lab}  {val}" for lab, val, _ in rows]) if title else \
            "\n".join(f"{lab}  {val}" for lab, val, _ in rows)
        ax = h["ax"]
        if self._tip is not None:
            try:
                self._tip.remove()
            except Exception:
                pass
        # 右端付近では左側に出す
        frac = (e.x - ax.bbox.x0) / max(ax.bbox.width, 1)
        dx = -14 if frac > 0.62 else 14
        ha = "right" if frac > 0.62 else "left"
        xy = (e.xdata, xs[i]) if h["axis"] == "y" else (xs[i], e.ydata)
        self._tip = ax.annotate(
            text, xy=xy, xycoords="data", xytext=(dx, 0), textcoords="offset points",
            ha=ha, va="center", fontsize=8.5, color=C["ink"], zorder=20, annotation_clip=False,
            bbox=dict(boxstyle="round,pad=0.55,rounding_size=0.3", fc="#ffffff", ec=C["border"], lw=1, alpha=0.97),
        )
        self._tip.set_in_layout(False)
        self.canvas.draw_idle()


class LabeledScale(tk.Frame):
    """ラベル＋値表示つきスライダー（resolution 刻みにスナップ）"""

    def __init__(self, parent, text, from_, to, resolution, value, fmt="{:.1f}", command=None, **kw):
        super().__init__(parent, bg=C["card"], **kw)
        self.res = resolution
        self.fmt = fmt
        self.command = command
        top = tk.Frame(self, bg=C["card"])
        top.pack(fill="x")
        tk.Label(top, text=text, font=FONTS["small"], fg=C["ink2"], bg=C["card"]).pack(side="left")
        self.val_label = tk.Label(top, text="", font=FONTS["small_b"], fg=C["ink"], bg=C["card"])
        self.val_label.pack(side="right")
        self.var = tk.DoubleVar(value=value)
        self.scale = ttk.Scale(self, from_=from_, to=to, variable=self.var, command=self._changed)
        self.scale.pack(fill="x", pady=(2, 0))
        self._changed()

    def _changed(self, *_):
        v = round(self.var.get() / self.res) * self.res
        self.val_label.configure(text=self.fmt.format(v))
        if self.command:
            self.command(v)

    def get(self) -> float:
        return round(self.var.get() / self.res) * self.res

    def set(self, v: float) -> None:
        self.var.set(v)
        self._changed()


class ScrollFrame(tk.Frame):
    """縦スクロール可能なフレーム（左側の操作パネル用）"""

    def __init__(self, parent, width=300, **kw):
        super().__init__(parent, bg=C["card"], **kw)
        self.canvas = tk.Canvas(self, bg=C["card"], highlightthickness=0, bd=0, width=width)
        self.vsb = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=C["card"])
        self._win = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.vsb.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vsb.pack(side="right", fill="y")
        self.inner.bind("<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfigure(self._win, width=e.width))
        self.inner.bind("<Enter>", lambda e: self._bind_wheel(True))
        self.inner.bind("<Leave>", lambda e: self._bind_wheel(False))

    def _bind_wheel(self, on):
        if on:
            self.canvas.bind_all("<MouseWheel>", self._wheel)
        else:
            self.canvas.unbind_all("<MouseWheel>")

    def _wheel(self, e):
        if self.inner.winfo_height() > self.canvas.winfo_height():
            self.canvas.yview_scroll(int(-e.delta / 120), "units")


class SectionLabel(tk.Label):
    def __init__(self, parent, text, **kw):
        super().__init__(parent, text=text, font=FONTS["small_b"], fg=C["ink2"], bg=C["card"], anchor="w", **kw)


class HoverTip:
    """ウィジェット用の簡易ツールチップ"""

    def __init__(self, widget, text):
        self.widget, self.text, self.tw = widget, text, None
        widget.bind("<Enter>", self.show)
        widget.bind("<Leave>", self.hide)

    def show(self, _e=None):
        if self.tw:
            return
        x = self.widget.winfo_rootx() + 12
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.tw = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(tw, text=self.text, font=FONTS["small"], bg="#ffffff", fg=C["ink"], justify="left",
                 relief="solid", bd=1, padx=8, pady=5, wraplength=theme.px(320)).pack()

    def hide(self, _e=None):
        if self.tw:
            self.tw.destroy()
            self.tw = None


def make_tree(parent, columns, widths, anchors=None, height=8):
    """Treeview（表）を作るヘルパ。columns: [(id, 見出し)]"""
    frame = tk.Frame(parent, bg=C["card"])
    tree = ttk.Treeview(frame, columns=[c for c, _ in columns], show="headings", height=height)
    for (cid, title), w in zip(columns, widths):
        tree.heading(cid, text=title)
        anchor = (anchors or {}).get(cid, "e")
        tree.column(cid, width=theme.px(w), anchor=anchor, stretch=True)
    vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=vsb.set)
    tree.pack(side="left", fill="both", expand=True)
    vsb.pack(side="right", fill="y")
    tree.tag_configure("odd", background="#fafaf8")
    return frame, tree
