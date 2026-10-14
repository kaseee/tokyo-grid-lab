"""配色・フォント・ttk スタイル・matplotlib 設定"""
from __future__ import annotations

import sys
from pathlib import Path

import tkinter as tk
from tkinter import font as tkfont
from tkinter import ttk

from core.datastore import asset

# ---------------------------------------------------------------------------
# 配色（ライトテーマ）
# ---------------------------------------------------------------------------
C = {
    "page": "#f3f3f0",
    "card": "#ffffff",
    "surface": "#fcfcfb",
    "sidebar": "#1d2026",
    "sidebar_hover": "#2a2e36",
    "sidebar_sel": "#323844",
    "sidebar_text": "#c9ccd3",
    "sidebar_muted": "#7d828d",
    "ink": "#0b0b0b",
    "ink2": "#52514e",
    "muted": "#898781",
    "grid": "#e8e7e1",
    "axis": "#c3c2b7",
    "border": "#e1e0d9",
    "accent": "#2a78d6",
    "accent_hover": "#256abf",
    "accent_soft": "#e8f1fc",
    "good_text": "#006300",
    "critical": "#d03b3b",
    "warning": "#fab219",
    "band_train": "#efeeea",
    "band_valid": "#e6e5df",
    "band_test": "#e3eefb",
}

# 系列色（検証済みカテゴリカルパレット：固定順）
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

# 特徴量グループ・日種別（先頭 3 色は全ペアで判別可能）
GROUP_COLOR = {"calendar": SERIES[0], "weather": SERIES[1], "lag": SERIES[2]}
DAYTYPE_COLOR = {0: SERIES[0], 1: SERIES[1], 2: SERIES[2]}
DAYTYPE_LABEL = {0: "平日", 1: "土曜", 2: "日曜・祝日"}

# 発電区分（積み上げ順で隣接ペアを検証済み）
DISPATCH_COLOR = {
    "base": "#6250d6",    # 原子力・水力
    "coal": "#eb6834",    # 石炭
    "lng": "#2a78d6",     # LNG 火力
    "oil": "#e34948",     # 石油・GT
    "import": "#008300",  # 連系線受電
    "pumped": "#1baf7a",  # 揚水発電
    "solar": "#eda100",   # 太陽光
}
POS_ERR = "#e34948"   # 予測が実績を上回る
NEG_ERR = "#2a78d6"   # 予測が実績を下回る

FONT_FAMILY = "Noto Sans CJK JP"
UI_SCALE = 1.0          # 画面の拡大率（Windows の表示スケール 150% → 1.5）
_IMAGES: list = []      # PhotoImage の参照保持用
_FALLBACK_FAMILIES = ["Yu Gothic UI", "Meiryo UI", "Yu Gothic", "Meiryo", "MS UI Gothic", "MS Gothic"]
FONTS: dict[str, tkfont.Font] = {}


def _font_files():
    d = asset("fonts")
    return [p for p in sorted(d.glob("*.otf"))] if d.exists() else []


def register_private_fonts() -> None:
    """同梱フォントをプロセス内だけで使えるように登録（Windows）。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        FR_PRIVATE = 0x10
        for p in _font_files():
            ctypes.windll.gdi32.AddFontResourceExW(str(p), FR_PRIVATE, 0)
    except Exception:
        pass


def px(n: float) -> int:
    """設計時（拡大率 100%）のピクセル値を現在の拡大率に合わせる"""
    return int(round(n * UI_SCALE))


def fit_scaling(root: tk.Tk, design_w=1440, design_h=860) -> float:
    """
    画面に収まるよう Tk の拡大率（tk scaling）を調整する。
    例）1920×1080 / 表示スケール 150% のノート PC では文字が大きすぎて画面に入らないため、
        全体を少し縮小して表示する（フォント・グラフ・チェックボックス画像が一緒に縮む）。
    """
    global UI_SCALE
    try:
        scaling = float(root.tk.call("tk", "scaling"))
    except Exception:
        scaling = 96 / 72
    base = 96 / 72
    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    need_w, need_h = design_w * scaling / base, design_h * scaling / base
    f = min(1.0, sw * 0.96 / need_w, sh * 0.90 / need_h)
    if f < 1.0:
        scaling = scaling * max(f, 0.7)
        root.tk.call("tk", "scaling", scaling)
    UI_SCALE = max(0.75, min(3.0, scaling / base))
    return UI_SCALE


def setup_fonts(root: tk.Tk) -> str:
    global FONT_FAMILY
    fams = set(tkfont.families(root))
    family = FONT_FAMILY if FONT_FAMILY in fams else next((f for f in _FALLBACK_FAMILIES if f in fams), "TkDefaultFont")
    FONT_FAMILY = family
    for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont", "TkCaptionFont", "TkTooltipFont"):
        try:
            tkfont.nametofont(name).configure(family=family, size=10)
        except tk.TclError:
            pass
    spec = {
        "body": (10, "normal"), "body_b": (10, "bold"), "small": (9, "normal"), "small_b": (9, "bold"),
        "tiny": (8, "normal"), "h1": (17, "bold"), "h2": (12, "bold"), "h3": (11, "bold"),
        "stat": (21, "bold"), "stat_s": (16, "bold"), "nav": (11, "normal"), "nav_b": (11, "bold"),
        "brand": (12, "bold"),
    }
    for k, (size, weight) in spec.items():
        FONTS[k] = tkfont.Font(root=root, family=family, size=size, weight=weight)
    return family


def _png_image(root, draw_fn, w, h, ss=4):
    """PIL で高解像度に描いて縮小 → PNG(base64) → tk.PhotoImage"""
    import base64
    import io

    from PIL import Image, ImageDraw
    big = Image.new("RGBA", (w * ss, h * ss), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(big), ss)
    img = big.resize((w, h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    ph = tk.PhotoImage(master=root, data=base64.b64encode(buf.getvalue()).decode("ascii"))
    _IMAGES.append(ph)
    return ph


def _hex(c, a=255):
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) + (a,)


def _custom_elements(root, st):
    n = max(14, round(15 * UI_SCALE))
    acc, brd, white = _hex(C["accent"]), _hex("#b9b8b0"), _hex("#ffffff")

    def box(on):
        def d(g, s):
            r = 3.2 * s
            if on:
                g.rounded_rectangle((s, s, n * s - s, n * s - s), radius=r, fill=acc)
                g.line([(3.6 * s * n / 15, 7.8 * s * n / 15), (6.4 * s * n / 15, 10.6 * s * n / 15),
                        (11.4 * s * n / 15, 4.8 * s * n / 15)], fill=white, width=int(1.9 * s * n / 15), joint="curve")
            else:
                g.rounded_rectangle((s, s, n * s - s, n * s - s), radius=r, fill=white, outline=brd, width=int(1.2 * s))
        return d

    def radio(on):
        def d(g, s):
            if on:
                g.ellipse((s, s, n * s - s, n * s - s), fill=acc)
                c = n * s / 2
                rr = 2.6 * s * n / 15
                g.ellipse((c - rr, c - rr, c + rr, c + rr), fill=white)
            else:
                g.ellipse((s, s, n * s - s, n * s - s), fill=white, outline=brd, width=int(1.2 * s))
        return d

    cb_off, cb_on = _png_image(root, box(False), n, n), _png_image(root, box(True), n, n)
    rb_off, rb_on = _png_image(root, radio(False), n, n), _png_image(root, radio(True), n, n)
    gap = n + max(6, round(7 * UI_SCALE))
    st.element_create("Fancy.Checkbutton.indicator", "image", cb_off, ("selected", cb_on), sticky="w", width=gap)
    st.element_create("Fancy.Radiobutton.indicator", "image", rb_off, ("selected", rb_on), sticky="w", width=gap)
    st.layout("TCheckbutton", [("Checkbutton.padding", {"sticky": "nswe", "children": [
        ("Fancy.Checkbutton.indicator", {"side": "left", "sticky": "w"}),
        ("Checkbutton.label", {"side": "left", "sticky": "nswe"})]})])
    st.layout("TRadiobutton", [("Radiobutton.padding", {"sticky": "nswe", "children": [
        ("Fancy.Radiobutton.indicator", {"side": "left", "sticky": "w"}),
        ("Radiobutton.label", {"side": "left", "sticky": "nswe"})]})])

    k = max(14, round(16 * UI_SCALE))
    tw = 3 * k

    def trough(g, s):
        y0, y1 = (k / 2 - 2.2 * UI_SCALE) * s, (k / 2 + 2.2 * UI_SCALE) * s
        g.rounded_rectangle((s, y0, tw * s - s, y1), radius=(y1 - y0) / 2, fill=_hex("#dcdbd4"))

    def slider(g, s):
        g.ellipse((s, s, k * s - s, k * s - s), fill=white, outline=acc, width=int(2.4 * s * UI_SCALE))

    tr = _png_image(root, trough, tw, k)
    sl = _png_image(root, slider, k, k)
    sl_a = _png_image(root, lambda g, s: (g.ellipse((s, s, k * s - s, k * s - s), fill=acc)), k, k)
    st.element_create("Fancy.Horizontal.Scale.trough", "image", tr, border=(k // 2, 0), sticky="ew")
    st.element_create("Fancy.Horizontal.Scale.slider", "image", sl, ("pressed", sl_a), ("active", sl_a), sticky="")
    st.layout("Horizontal.TScale", [("Fancy.Horizontal.Scale.trough", {"sticky": "ew", "children": [
        ("Fancy.Horizontal.Scale.slider", {"side": "left", "sticky": ""})]})])


def setup_ttk(root: tk.Tk) -> None:
    st = ttk.Style(root)
    st.theme_use("clam")
    f = FONTS
    st.configure(".", background=C["card"], foreground=C["ink"], font=f["body"], bordercolor=C["border"],
                 lightcolor=C["card"], darkcolor=C["card"], troughcolor=C["grid"], focuscolor=C["accent"])
    st.configure("TFrame", background=C["card"])
    st.configure("Page.TFrame", background=C["page"])
    st.configure("TLabel", background=C["card"], foreground=C["ink"])
    st.configure("Muted.TLabel", foreground=C["muted"], font=f["small"])
    st.configure("Ink2.TLabel", foreground=C["ink2"])
    st.configure("H2.TLabel", font=f["h2"])
    st.configure("H3.TLabel", font=f["h3"])
    st.configure("Page.TLabel", background=C["page"])

    st.configure("TButton", padding=(12, 6), relief="flat", background="#f1f0ec", foreground=C["ink"],
                 bordercolor=C["border"], font=f["body"])
    st.map("TButton", background=[("disabled", "#f4f4f2"), ("pressed", "#e4e3de"), ("active", "#e9e8e3")],
           foreground=[("disabled", C["muted"])])
    st.configure("Accent.TButton", background=C["accent"], foreground="#ffffff", bordercolor=C["accent"],
                 font=f["body_b"], padding=(14, 8))
    st.map("Accent.TButton", background=[("disabled", "#9dbfe9"), ("pressed", "#1c5cab"), ("active", C["accent_hover"])],
           foreground=[("disabled", "#f4f8fd")])
    st.configure("Chip.TButton", padding=(8, 3), font=f["small"], background="#f1f0ec")

    st.configure("TCheckbutton", background=C["card"], foreground=C["ink"], font=f["body"])
    st.map("TCheckbutton", background=[("active", C["card"])], indicatorcolor=[("selected", C["accent"]), ("!selected", "#ffffff")])
    st.configure("TRadiobutton", background=C["card"], foreground=C["ink"], font=f["body"])
    st.map("TRadiobutton", background=[("active", C["card"])], indicatorcolor=[("selected", C["accent"]), ("!selected", "#ffffff")])

    st.configure("Horizontal.TScale", background=C["card"], troughcolor="#e6e5df", sliderthickness=14,
                 bordercolor=C["border"], lightcolor=C["accent"], darkcolor=C["accent"])
    st.configure("TCombobox", padding=4, arrowsize=14, fieldbackground="#ffffff", background="#f6f6f3",
                 bordercolor=C["border"], lightcolor="#ffffff", darkcolor="#ffffff", arrowcolor=C["ink2"])
    st.map("TCombobox", fieldbackground=[("readonly", "#ffffff")], background=[("active", "#ebeae5")],
           selectbackground=[("readonly", "#ffffff")], selectforeground=[("readonly", C["ink"])])
    st.configure("TSpinbox", padding=3, arrowsize=12, fieldbackground="#ffffff")

    st.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", foreground=C["ink"], rowheight=px(24),
                 font=f["small"], bordercolor=C["border"], borderwidth=0)
    st.configure("Treeview.Heading", background="#f4f4f1", foreground=C["ink2"], font=f["small_b"], relief="flat",
                 bordercolor=C["border"], padding=(6, 4))
    st.map("Treeview", background=[("selected", C["accent_soft"])], foreground=[("selected", C["ink"])])
    st.map("Treeview.Heading", background=[("active", "#ebeae5")])

    st.configure("TNotebook", background=C["card"], borderwidth=0, tabmargins=(0, 0, 0, 0))
    st.configure("TNotebook", bordercolor=C["border"], lightcolor=C["card"], darkcolor=C["card"])
    st.configure("TNotebook.Tab", background="#f4f4f1", foreground=C["ink2"], padding=(14, 6), borderwidth=0,
                 font=f["body"], bordercolor=C["border"], lightcolor="#f4f4f1", darkcolor="#f4f4f1")
    st.map("TNotebook.Tab", background=[("selected", C["card"]), ("active", "#ecebe6")],
           foreground=[("selected", C["accent"]), ("active", C["ink"])],
           lightcolor=[("selected", C["card"])], darkcolor=[("selected", C["card"])],
           font=[("selected", f["body_b"])], expand=[("selected", (0, 0, 0, 0))])
    st.configure("Horizontal.TProgressbar", troughcolor="#e6e5df", background=C["accent"], bordercolor=C["card"],
                 lightcolor=C["accent"], darkcolor=C["accent"], thickness=6)
    st.configure("TSeparator", background=C["border"])
    st.configure("Vertical.TScrollbar", background="#e6e5df", troughcolor=C["card"], bordercolor=C["card"],
                 lightcolor="#e6e5df", darkcolor="#e6e5df", arrowcolor=C["muted"], gripcount=0, arrowsize=12)
    try:
        _custom_elements(root, st)
    except Exception as e:  # 画像要素が作れない環境では標準の見た目のまま
        print("custom ttk elements unavailable:", e, file=sys.stderr)


def setup_matplotlib(family: str) -> None:
    import matplotlib
    from matplotlib import font_manager

    for p in _font_files():
        try:
            font_manager.fontManager.addfont(str(p))
        except Exception:
            pass
    fams = [family, "Noto Sans CJK JP", "Yu Gothic", "Meiryo", "MS Gothic", "IPAexGothic", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    seen, fam = set(), []
    for x in fams:
        if x not in seen and x in available:
            fam.append(x)
            seen.add(x)
    if "DejaVu Sans" not in fam:
        fam.append("DejaVu Sans")
    rc = matplotlib.rcParams
    rc.update({
        "font.family": fam,
        "font.size": 9,
        "axes.unicode_minus": False,
        "figure.facecolor": C["card"],
        "axes.facecolor": C["card"],
        "axes.edgecolor": C["axis"],
        "axes.labelcolor": C["ink2"],
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlecolor": C["ink"],
        "axes.titlepad": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": C["grid"],
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": C["axis"],
        "ytick.color": C["card"],
        "xtick.labelcolor": C["muted"],
        "ytick.labelcolor": C["muted"],
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "ytick.major.size": 0,
        "xtick.major.size": 3,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "legend.labelcolor": C["ink2"],
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
        "path.simplify": True,
    })
