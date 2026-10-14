"""
東京エリア 電力需要予測 × 発電計画最適化（可視化アプリ）

  需要予測    : 東京電力PG「でんき予報」実績 ＋ 気象庁 時別値 → LightGBM（MAPE で評価）
  発電計画    : PuLP で起動停止つき経済負荷配分（MILP）を定式化 → HiGHS で最小コスト求解
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

APP_TITLE = "東京エリア 電力需要予測 × 発電計画最適化"
APP_VERSION = "1.0.0"


def _dpi_aware():
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass


def _close_splash():
    try:
        import pyi_splash  # PyInstaller のスプラッシュ画面
        pyi_splash.close()
    except Exception:
        pass


def main():
    _dpi_aware()
    import tkinter as tk
    from tkinter import messagebox

    import matplotlib
    matplotlib.use("TkAgg")

    from core.datastore import DataStore, asset
    from ui import theme
    from ui.theme import C, FONTS

    theme.register_private_fonts()
    root = tk.Tk()
    root.withdraw()
    root.title(APP_TITLE)
    try:
        if sys.platform == "win32":
            root.iconbitmap(default=str(asset("app.ico")))
        else:
            root.iconphoto(True, tk.PhotoImage(file=str(asset("app_icon.png"))))
    except Exception:
        pass
    scale = theme.fit_scaling(root)
    family = theme.setup_fonts(root)
    theme.setup_ttk(root)
    theme.setup_matplotlib(family)

    def report(exc, val, tb):
        msg = "".join(traceback.format_exception(exc, val, tb))
        try:
            log = Path(os.environ.get("TEMP", ".")) / "tepco_app_error.log"
            log.write_text(msg, encoding="utf-8")
        except Exception:
            pass
        messagebox.showerror("エラー", f"予期しないエラーが発生しました。\n\n{val}")

    root.report_callback_exception = report

    ds = DataStore()
    app = App(root, ds)

    sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
    w, h = min(int(1480 * scale), int(sw * 0.94)), min(int(900 * scale), int(sh * 0.88))
    root.geometry(f"{w}x{h}+{max((sw - w) // 2, 0)}+{max((sh - h) // 3, 0)}")
    root.minsize(min(int(1100 * scale), sw - 40), min(int(660 * scale), sh - 80))
    root.configure(bg=C["page"])
    root.deiconify()
    root.after(200, _close_splash)
    if os.environ.get("TEPCO_APP_PAGE"):  # スクリーンショット用（開発時のみ使用）
        def _goto():
            app.show(os.environ["TEPCO_APP_PAGE"])
            tab = os.environ.get("TEPCO_APP_TAB")
            page = app.pages.get(app.current)
            if tab and page is not None and hasattr(page, "nb"):
                root.after(2500, lambda: page.nb.select(int(tab)))
        root.after(300, _goto)
    root.mainloop()


class App:
    def __init__(self, root, ds):
        import tkinter as tk

        from ui.page_about import AboutPage
        from ui.page_dispatch import DispatchPage
        from ui.page_forecast import ForecastPage
        from ui.page_overview import OverviewPage
        from ui.theme import C, FONTS

        self.root, self.ds = root, ds
        self.page_classes = {"overview": OverviewPage, "forecast": ForecastPage, "dispatch": DispatchPage,
                             "about": AboutPage}
        self.pages = {}
        self.current = None

        # ---- サイドバー
        from ui.theme import px
        side = tk.Frame(root, bg=C["sidebar"], width=px(214))
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        brand = tk.Frame(side, bg=C["sidebar"])
        brand.pack(fill="x", padx=18, pady=(20, 22))
        k = px(28) / 28
        mark = tk.Canvas(brand, width=px(28), height=px(28), bg=C["sidebar"], highlightthickness=0)
        mark.pack(side="left", anchor="n")
        mark.create_rectangle(2 * k, 16 * k, 8 * k, 26 * k, fill="#6250d6", width=0)
        mark.create_rectangle(11 * k, 9 * k, 17 * k, 26 * k, fill="#2a78d6", width=0)
        mark.create_rectangle(20 * k, 3 * k, 26 * k, 26 * k, fill="#eda100", width=0)
        bt = tk.Frame(brand, bg=C["sidebar"])
        bt.pack(side="left", padx=(10, 0))
        tk.Label(bt, text="Tokyo Grid Lab", font=FONTS["brand"], fg="#ffffff", bg=C["sidebar"]).pack(anchor="w")
        tk.Label(bt, text="需要予測 × 発電計画", font=FONTS["tiny"], fg=C["sidebar_muted"], bg=C["sidebar"]).pack(anchor="w")

        self.nav = {}
        for key, label, desc in [("overview", "概要", "結果のサマリー"),
                                 ("forecast", "需要予測", "LightGBM ・ MAPE"),
                                 ("dispatch", "発電計画最適化", "PuLP ＋ HiGHS"),
                                 ("about", "データと手法", "出典・前提")]:
            self.nav[key] = NavItem(side, label, desc, lambda k=key: self.show(k))
            self.nav[key].pack(fill="x")
        foot = tk.Frame(side, bg=C["sidebar"])
        foot.pack(side="bottom", fill="x", padx=18, pady=16)
        p = ds.metrics["periods"]
        tk.Label(foot, text=f"データ {p['data_start'][:7].replace('-', '/')}〜{p['data_end'][:10].replace('-', '/')}",
                 font=FONTS["tiny"], fg=C["sidebar_muted"], bg=C["sidebar"]).pack(anchor="w")
        tk.Label(foot, text=f"v{APP_VERSION} ・ 公開データによる個人作品", font=FONTS["tiny"], fg=C["sidebar_muted"],
                 bg=C["sidebar"], wraplength=px(176), justify="left").pack(anchor="w")

        # ---- メイン
        main = tk.Frame(root, bg=C["page"])
        main.pack(side="left", fill="both", expand=True)
        head = tk.Frame(main, bg=C["page"])
        head.pack(fill="x", padx=24, pady=(18, 12))
        self.h_title = tk.Label(head, text="", font=FONTS["h1"], fg=C["ink"], bg=C["page"])
        self.h_title.pack(anchor="w")
        self.h_sub = tk.Label(head, text="", font=FONTS["body"], fg=C["ink2"], bg=C["page"])
        self.h_sub.pack(anchor="w", pady=(2, 0))
        self.container = tk.Frame(main, bg=C["page"])
        self.container.pack(fill="both", expand=True, padx=24, pady=(0, 20))
        self.show("overview")

    def show(self, key):
        if key not in self.page_classes:
            return
        if key not in self.pages:
            self.root.config(cursor="watch")
            self.root.update_idletasks()
            self.pages[key] = self.page_classes[key](self.container, self)
            self.root.config(cursor="")
        if self.current:
            self.pages[self.current].pack_forget()
            self.nav[self.current].set_selected(False)
        page = self.pages[key]
        page.pack(fill="both", expand=True)
        self.nav[key].set_selected(True)
        self.h_title.configure(text=page.title)
        self.h_sub.configure(text=page.subtitle)
        self.current = key
        if hasattr(page, "on_show"):
            self.root.after(80, page.on_show)

    def on_forecast_changed(self):
        if "dispatch" in self.pages:
            self.pages["dispatch"].on_forecast_changed()


class NavItem:
    def __init__(self, parent, label, desc, command):
        import tkinter as tk

        from ui.theme import C, FONTS
        self.C = C
        self.frame = tk.Frame(parent, bg=C["sidebar"], cursor="hand2")
        self.bar = tk.Frame(self.frame, bg=C["sidebar"], width=4)
        self.bar.pack(side="left", fill="y")
        inner = tk.Frame(self.frame, bg=C["sidebar"])
        inner.pack(side="left", fill="x", padx=(14, 10), pady=9)
        self.l1 = tk.Label(inner, text=label, font=FONTS["nav"], fg=C["sidebar_text"], bg=C["sidebar"], anchor="w")
        self.l1.pack(anchor="w")
        self.l2 = tk.Label(inner, text=desc, font=FONTS["tiny"], fg=C["sidebar_muted"], bg=C["sidebar"], anchor="w")
        self.l2.pack(anchor="w")
        self.inner = inner
        self.selected = False
        self.fonts = FONTS
        for w in (self.frame, inner, self.l1, self.l2):
            w.bind("<Button-1>", lambda e: command())
            w.bind("<Enter>", lambda e: self._hover(True))
            w.bind("<Leave>", lambda e: self._hover(False))

    def pack(self, **kw):
        self.frame.pack(**kw)

    def _paint(self, bg):
        for w in (self.frame, self.inner, self.l1, self.l2):
            w.configure(bg=bg)

    def _hover(self, on):
        if self.selected:
            return
        self._paint(self.C["sidebar_hover"] if on else self.C["sidebar"])

    def set_selected(self, sel):
        self.selected = sel
        self._paint(self.C["sidebar_sel"] if sel else self.C["sidebar"])
        self.bar.configure(bg="#5aa0ff" if sel else self.C["sidebar"])
        self.l1.configure(fg="#ffffff" if sel else self.C["sidebar_text"],
                          font=self.fonts["nav_b"] if sel else self.fonts["nav"])


if __name__ == "__main__":
    main()
