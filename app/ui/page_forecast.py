"""需要予測ページ（LightGBM）"""
from __future__ import annotations

import queue
import threading
import time
import tkinter as tk
from tkinter import ttk

import numpy as np

from ui import charts
from ui.theme import C, FONTS, GROUP_COLOR, px
from ui.widgets import Card, ChartCanvas, HoverTip, ScrollFrame, SectionLabel, StatTile, make_tree

WEEK = "月火水木金土日"
GROUP_JA = {"calendar": "カレンダー", "weather": "気象", "lag": "需要ラグ"}


def _dstr(d) -> str:
    d = np.datetime64(d, "D").astype(object)
    return f"{d.year}/{d.month:02d}/{d.day:02d}（{WEEK[d.weekday()]}）"


class ForecastPage(tk.Frame):
    title = "需要予測"
    subtitle = "LightGBM による東京エリア 1 時間値の前日予測（テスト期間は学習に未使用）"

    def __init__(self, parent, app):
        super().__init__(parent, bg=C["page"])
        self.app = app
        ds = app.ds
        self.cur = dict(met=ds.metrics["test"], importance=self._imp_items(ds.metrics["importance_gain"]),
                        best_iter=ds.metrics["model"]["best_iter"], secs=ds.metrics["model"]["train_seconds"],
                        label="初期モデル")
        self.q: queue.Queue = queue.Queue()
        self.training = False

        # ---- 指標タイル
        tiles = tk.Frame(self, bg=C["page"])
        tiles.pack(fill="x", pady=(0, 12))
        self.t_mape = StatTile(tiles, "MAPE（平均絶対誤差率）", "-", "")
        self.t_peak = StatTile(tiles, "日最大需要の MAPE", "-", "")
        self.t_mae = StatTile(tiles, "MAE（平均絶対誤差）", "-", "")
        self.t_r2 = StatTile(tiles, "決定係数 R²", "-", "")
        self.t_model = StatTile(tiles, "現在のモデル", "-", "")
        for i, t in enumerate((self.t_mape, self.t_peak, self.t_mae, self.t_r2, self.t_model)):
            t.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 10, 0))
            tiles.grid_columnconfigure(i, weight=1, uniform="t")

        main = tk.Frame(self, bg=C["page"])
        main.pack(fill="both", expand=True)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(0, weight=1)

        # ---- グラフ（タブ）
        left = Card(main, pad=10)
        left.grid(row=0, column=0, sticky="nsew")
        self.nb = ttk.Notebook(left.body)
        self.nb.pack(fill="both", expand=True)
        self.tabs = {}
        for key, lab in [("ts", "実績と予測（時系列）"), ("err", "月別・時間帯別の誤差"), ("sc", "散布図"),
                         ("imp", "特徴量の重要度"), ("temp", "気温と需要"), ("abl", "特徴量の比較")]:
            f = tk.Frame(self.nb, bg=C["card"])
            self.nb.add(f, text=lab)
            self.tabs[key] = f
        self._build_ts_tab()
        self.charts = {}
        for key in ("err", "sc", "imp", "temp", "abl"):
            ch = ChartCanvas(self.tabs[key], height=4.2)
            ch.pack(fill="both", expand=True, pady=(8, 0))
            self.charts[key] = ch
        self.nb.bind("<<NotebookTabChanged>>", lambda e: self._render_current())

        # ---- 再学習パネル
        right = Card(main, pad=0)
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        sf = ScrollFrame(right.body, width=px(318))
        sf.pack(fill="both", expand=True)
        inner = tk.Frame(sf.inner, bg=C["card"])
        inner.pack(fill="both", expand=True, padx=16, pady=14)
        tk.Label(inner, text="モデルの再学習", font=FONTS["h3"], bg=C["card"], fg=C["ink"]).pack(anchor="w", pady=(0, 6))
        self._build_train_panel(inner)

        self.update_tiles()
        self._dirty = set(self.tabs)
        self.after(50, self._render_current)
        self.after(150, self._poll)

    # ------------------------------------------------------------------
    @staticmethod
    def _imp_items(raw):
        from core.features_meta import GROUP_OF
        return [dict(name=r["name"], label=r["label"], value=r["value"], group=GROUP_OF.get(r["name"], "calendar"))
                for r in raw]

    def _build_ts_tab(self):
        f = self.tabs["ts"]
        ctrl = tk.Frame(f, bg=C["card"])
        ctrl.pack(fill="x", pady=(10, 2))
        ds = self.app.ds
        mondays = [d for d in ds.test_days if np.datetime64(d, "D").astype(object).weekday() == 0]
        self.week_starts = mondays
        self.week_var = tk.StringVar()
        tk.Label(ctrl, text="表示開始日", font=FONTS["small"], bg=C["card"], fg=C["ink2"]).pack(side="left")
        ttk.Button(ctrl, text="◀", width=3, style="Chip.TButton", command=lambda: self._shift(-1)).pack(side="left", padx=(8, 2))
        self.week_cb = ttk.Combobox(ctrl, textvariable=self.week_var, state="readonly", width=17,
                                    values=[_dstr(d) for d in mondays])
        self.week_cb.pack(side="left")
        ttk.Button(ctrl, text="▶", width=3, style="Chip.TButton", command=lambda: self._shift(1)).pack(side="left", padx=(2, 14))
        self.week_cb.bind("<<ComboboxSelected>>", lambda e: self._render_ts())
        tk.Label(ctrl, text="表示期間", font=FONTS["small"], bg=C["card"], fg=C["ink2"]).pack(side="left")
        self.span = tk.IntVar(value=7)
        for n, lab in [(7, "1週間"), (14, "2週間"), (28, "4週間")]:
            ttk.Radiobutton(ctrl, text=lab, value=n, variable=self.span, command=self._render_ts).pack(side="left", padx=(6, 0))
        self.show_tepco = tk.BooleanVar(value=False)
        cb = ttk.Checkbutton(ctrl, text="東電の当日予測（参考）", variable=self.show_tepco, command=self._render_ts)
        cb.pack(side="left", padx=(16, 0))
        HoverTip(cb, "でんき予報の「予測値」。当日中に随時更新される予測のため、前日予測とは条件が異なります（参考表示）。")
        # 既定は直近の夏の最大需要日を含む週
        nd = ds.notable_days().get("夏の最大需要日")
        idx = 0
        if nd is not None:
            idx = max(0, int(np.searchsorted(np.array(mondays, dtype="datetime64[D]"), np.datetime64(nd, "D"), side="right")) - 1)
        self.week_cb.current(idx)
        self.ts_chart = ChartCanvas(f, height=4.0)
        self.ts_chart.pack(fill="both", expand=True)
        self.ts_note = tk.Label(f, text="", font=FONTS["small"], bg=C["card"], fg=C["muted"], anchor="w", justify="left")
        self.ts_note.pack(fill="x", pady=(2, 0))

    def _shift(self, k):
        i = self.week_cb.current() + k * max(1, self.span.get() // 7)
        i = int(np.clip(i, 0, len(self.week_starts) - 1))
        self.week_cb.current(i)
        self._render_ts()

    def _build_train_panel(self, body):
        intro = tk.Label(body, text="特徴量やハイパーパラメータを変えて、その場で LightGBM を学習し直します。"
                                    "再学習した予測は「発電計画最適化」にも反映されます。",
                         font=FONTS["small"], bg=C["card"], fg=C["ink2"], justify="left", anchor="w", wraplength=260)
        intro.pack(fill="x")
        body.bind("<Configure>", lambda e: intro.configure(wraplength=max(e.width - 8, 120)), add="+")
        SectionLabel(body, "使用する特徴量").pack(fill="x", pady=(12, 2))
        self.grp_vars = {}
        ds = self.app.ds
        for g in ("calendar", "weather", "lag"):
            n = sum(1 for x in ds.feature_groups if x == g)
            v = tk.BooleanVar(value=True)
            row = tk.Frame(body, bg=C["card"])
            row.pack(fill="x")
            tk.Frame(row, bg=GROUP_COLOR[g], width=10, height=10).pack(side="left", padx=(2, 6))
            ttk.Checkbutton(row, text=f"{GROUP_JA[g]}（{n}個）", variable=v).pack(side="left")
            self.grp_vars[g] = v

        SectionLabel(body, "ハイパーパラメータ").pack(fill="x", pady=(12, 2))
        grid = tk.Frame(body, bg=C["card"])
        grid.pack(fill="x")
        self.lr = tk.StringVar(value="0.05")
        self.leaves = tk.StringVar(value="31")
        self.rounds = tk.StringVar(value="4000")
        for r, (lab, var, vals) in enumerate([("学習率", self.lr, ["0.02", "0.05", "0.1", "0.2"]),
                                              ("葉の数（num_leaves）", self.leaves, ["15", "31", "63", "127"]),
                                              ("最大ラウンド数", self.rounds, ["500", "1000", "2000", "4000"])]):
            tk.Label(grid, text=lab, font=FONTS["small"], bg=C["card"], fg=C["ink2"]).grid(row=r, column=0, sticky="w", pady=3)
            ttk.Combobox(grid, textvariable=var, values=vals, width=8, state="readonly").grid(row=r, column=1, sticky="e", pady=3)
        grid.grid_columnconfigure(0, weight=1)
        self.log_target = tk.BooleanVar(value=True)
        cbl = ttk.Checkbutton(body, text="需要を対数変換して学習", variable=self.log_target)
        cbl.pack(anchor="w", pady=(6, 0))
        HoverTip(cbl, "log(需要) を二乗誤差で学習すると、相対誤差（MAPE）に近い基準で木が作られます。")

        self.btn_train = ttk.Button(body, text="再学習して評価", style="Accent.TButton", command=self.start_training)
        self.btn_train.pack(fill="x", pady=(14, 6))
        self.prog = ttk.Progressbar(body, mode="determinate", maximum=100)
        self.prog.pack(fill="x")
        self.train_status = tk.Label(body, text="学習期間 2022/4〜2025/9（early stopping 用に 2025/4〜9 を検証に使用）",
                                     font=FONTS["small"], bg=C["card"], fg=C["muted"], justify="left", anchor="w",
                                     wraplength=260)
        self.train_status.pack(fill="x", pady=(4, 0))
        body.bind("<Configure>", lambda e: self.train_status.configure(wraplength=max(e.width - 8, 120)), add="+")

        SectionLabel(body, "学習履歴（テスト MAPE）").pack(fill="x", pady=(14, 4))
        cols = [("no", "#"), ("feat", "特徴量"), ("lr", "学習率"), ("lv", "葉"), ("it", "木"), ("mape", "MAPE"), ("sec", "秒")]
        fr, self.hist = make_tree(body, cols, [26, 92, 44, 34, 44, 54, 36], anchors={"feat": "w", "no": "center"},
                                  height=6)
        fr.pack(fill="both", expand=True)
        m0 = self.app.ds.metrics
        self._hist_n = 0
        self._add_hist("全部", "0.05", "31", m0["model"]["best_iter"], m0["test"]["MAPE"], m0["model"]["train_seconds"])
        ttk.Button(body, text="初期モデルに戻す", command=self.reset_model).pack(fill="x", pady=(8, 0))

    def _add_hist(self, feat, lr, lv, it, mape, sec):
        self._hist_n += 1
        tag = ("odd",) if self._hist_n % 2 else ()
        self.hist.insert("", 0, values=(self._hist_n, feat, lr, lv, it, f"{mape:.3f}%", f"{sec:.0f}"), tags=tag)

    # ------------------------------------------------------------------
    def update_tiles(self):
        m = self.cur["met"]
        ds = self.app.ds
        naive = ds.metrics["baselines"]["naive_week"]
        base = ds.metrics["test"]["MAPE"]
        diff = m["MAPE"] - base
        sub = "初期モデルと同じ" if abs(diff) < 1e-4 else f"初期モデル比 {diff:+.3f} pt"
        self.t_mape.set(f"{m['MAPE']:.2f}%", f"{sub} ／ 前週同時刻法 {naive['MAPE']:.2f}%")
        self.t_peak.set(f"{m['peak_MAPE']:.2f}%", f"前週同時刻法 {naive['peak_MAPE']:.2f}%")
        self.t_mae.set(f"{m['MAE']:.0f} 万kW", f"RMSE {m['RMSE']:.0f} 万kW ／ 平均バイアス {m['bias']:+.0f} 万kW")
        self.t_r2.set(f"{m['R2']:.4f}", f"テスト {m['n']:,} 時間（2025/10〜2026/10）")
        self.t_model.set(f"{self.cur['best_iter']:,} 本", f"{self.cur['label']} ／ 学習 {self.cur['secs']:.0f} 秒")

    def _render_current(self):
        tab = self.nb.index(self.nb.select())
        key = list(self.tabs)[tab]
        if key == "ts":
            self._render_ts()
        elif key in self._dirty:
            getattr(self, f"_render_{key}")()
        self._dirty.discard(key)

    def _render_ts(self):
        ds = self.app.ds
        ch = self.ts_chart
        ch.clear()
        gs = ch.fig.add_gridspec(2, 1, height_ratios=[3, 1.1])
        ax = ch.fig.add_subplot(gs[0])
        ax2 = ch.fig.add_subplot(gs[1], sharex=ax)
        start = self.week_starts[self.week_cb.current()]
        n = self.span.get()
        r = charts.plot_forecast_window(ax, ax2, ds, start, n, show_tepco=self.show_tepco.get())
        ts, y, p, err, tep, temp = r["ts"], r["y"], r["p"], r["err"], r["tepco"], r["temp"]

        def rows(i):
            out = [("実績", f"{y[i]:,.0f} 万kW", None), ("予測", f"{p[i]:,.0f} 万kW", None),
                   ("誤差", f"{err[i]:+.1f}%", None), ("気温", f"{temp[i]:.1f}℃", None)]
            if self.show_tepco.get():
                out.insert(2, ("東電当日予測", f"{tep[i]:,.0f} 万kW", None))
            return out

        def title(i):
            d = ts[i].astype(object)
            return f"{d.month}/{d.day}（{WEEK[d.weekday()]}）{d.hour}:00〜{d.hour + 1}:00"

        ch.add_hover(ax, r["x"], rows, title, span_axes=[ax, ax2])
        ch.draw()
        ape = np.abs(err)
        self.ts_note.configure(
            text=f"表示期間の MAPE {ape.mean():.2f}% ／ 最大誤差 {ape.max():.1f}%　　"
                 "※ 気象は実測値を「完全な予報」として使用（前日予測の想定）。背景の灰色は土曜・日曜祝日。")

    def _render_err(self):
        ds = self.app.ds
        ch = self.charts["err"]
        ch.clear()
        ax1 = ch.fig.add_subplot(121)
        ax2 = ch.fig.add_subplot(122)
        months, vals = ds.monthly_mape()
        x1 = charts.plot_monthly_mape(ax1, months, vals, self.cur["met"]["MAPE"])
        hv = ds.hourly_mape()
        x2 = charts.plot_hourly_mape(ax2, hv)
        mo = months.astype(object)
        ch.add_hover(ax1, x1, lambda i: [("MAPE", f"{vals[i]:.2f}%", None)],
                     lambda i: f"{mo[i].year}年{mo[i].month}月", vline=False)
        ch.add_hover(ax2, x2, lambda i: [("MAPE", f"{hv[i]:.2f}%", None)], lambda i: f"{i}:00〜{i + 1}:00", vline=False)
        ch.draw()

    def _render_sc(self):
        ds = self.app.ds
        ch = self.charts["sc"]
        ch.clear()
        ax = ch.fig.add_subplot(111)
        t = ds.m_test
        charts.plot_scatter(ax, ds.demand[t], ds.pred[t], self.cur["met"])
        ch.draw()

    def _render_imp(self):
        ch = self.charts["imp"]
        ch.clear()
        ax = ch.fig.add_subplot(111)
        items = self.cur["importance"]
        charts.plot_importance(ax, items)
        top = items[:15][::-1]
        ch.add_hover(ax, np.arange(len(top)), lambda i: [("重要度", f"{top[i]['value']:.2f}%", None),
                                                      ("グループ", GROUP_JA[top[i]['group']], None)],
                     lambda i: top[i]["label"], vline=False, axis="y")
        ch.draw()

    def _render_temp(self):
        ds = self.app.ds
        ch = self.charts["temp"]
        ch.clear()
        ax = ch.fig.add_subplot(111)
        m = ds.m_train | ds.m_test
        charts.plot_temp_demand(ax, ds.temp[m], ds.demand[m], ds.daytype[m])
        ch.draw()

    def _render_abl(self):
        ds = self.app.ds
        met = ds.metrics
        ch = self.charts["abl"]
        ch.clear()
        ax = ch.fig.add_subplot(111)
        lab = {"calendar": "カレンダーのみ", "calendar+lag": "カレンダー＋需要ラグ",
               "calendar+weather": "カレンダー＋気象", "all": "全特徴量（本モデル）"}
        rows = [("前週同時刻（比較用）", met["baselines"]["naive_week"]["MAPE"], False)]
        rows += [(lab[a["key"]], a["MAPE"], a["key"] == "all") for a in met["ablation"]]
        charts.plot_ablation(ax, rows)
        tep = met["baselines"]["tepco_intraday"]["MAPE"]
        ax.text(0, -0.16, f"参考：東電の当日予測値（当日中に更新される予測で前日予測とは条件が異なる）の MAPE は {tep:.2f}%",
                transform=ax.transAxes, fontsize=8.5, color=C["muted"])
        ch.draw()

    # ------------------------------------------------------------------
    def start_training(self):
        if self.training:
            return
        groups = [g for g, v in self.grp_vars.items() if v.get()]
        if not groups:
            self.train_status.configure(text="特徴量グループを1つ以上選んでください。", fg=C["critical"])
            return
        ds = self.app.ds
        cols = [i for i, g in enumerate(ds.feature_groups) if g in groups]
        params = {"learning_rate": float(self.lr.get()), "num_leaves": int(self.leaves.get()), "min_data_in_leaf": 40}
        rounds = int(self.rounds.get())
        log_t = self.log_target.get()
        self.training = True
        self.btn_train.state(["disabled"])
        self.prog.configure(value=0)
        self.train_status.configure(text="LightGBM を読み込み中…", fg=C["ink2"])
        self._train_info = dict(groups=groups, params=params, rounds=rounds, log=log_t)
        threading.Thread(target=self._train_worker, args=(cols, params, rounds, log_t), daemon=True).start()

    def _train_worker(self, cols, params, rounds, log_t):
        try:
            from core.forecast import importance, predict, train_lgbm
            ds = self.app.ds
            phase = {"n": 1}

            def prog(it, end):
                self.q.put(("progress", phase["n"], it, end))

            t0 = time.perf_counter()
            res = train_lgbm(ds.X, ds.demand, ds.feature_names, cols, ds.m_fit, ds.m_valid, refit_mask=None,
                             params=params, num_boost_round=rounds, log_target=log_t, progress=prog)
            phase["n"] = 2
            self.q.put(("progress", 2, 0, int(res.best_iter * 1.1)))
            res2 = train_lgbm(ds.X, ds.demand, ds.feature_names, cols, ds.m_train, None, refit_mask=None,
                              params=params, num_boost_round=int(res.best_iter * 1.1), log_target=log_t,
                              progress=prog)
            pred = predict(res2, ds.X, cols, log_target=log_t)
            imp = importance(res2.booster)
            self.q.put(("done", pred, imp, int(res.best_iter * 1.1), time.perf_counter() - t0))
        except Exception as e:  # noqa: BLE001
            import traceback
            self.q.put(("error", f"{e}\n{traceback.format_exc(limit=3)}"))

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "progress":
                    _, ph, it, end = msg
                    frac = min(it / max(end, 1), 1.0)
                    self.prog.configure(value=(frac * 70 if ph == 1 else 70 + frac * 30))
                    txt = "① early stopping で木の本数を決定中" if ph == 1 else "② 学習期間全体で再学習中"
                    self.train_status.configure(text=f"{txt}… {it:,} 本", fg=C["ink2"])
                elif msg[0] == "done":
                    self._on_trained(*msg[1:])
                elif msg[0] == "error":
                    self.training = False
                    self.btn_train.state(["!disabled"])
                    self.train_status.configure(text="学習に失敗しました：" + msg[1][:300], fg=C["critical"])
        except queue.Empty:
            pass
        self.after(120, self._poll)

    def _on_trained(self, pred, imp, best, secs):
        from core.features_meta import GROUP_OF, LABEL_OF
        ds = self.app.ds
        ds.pred = np.asarray(pred, float)
        met = ds.test_metrics()
        info = self._train_info
        feat = "全部" if len(info["groups"]) == 3 else "＋".join(GROUP_JA[g][:2] for g in info["groups"])
        items = [dict(name=n, label=LABEL_OF.get(n, n), value=v, group=GROUP_OF.get(n, "calendar")) for n, v in imp]
        self.cur = dict(met=met, importance=items, best_iter=best, secs=secs, label="再学習モデル")
        self._add_hist(feat, f"{info['params']['learning_rate']:g}", str(info["params"]["num_leaves"]), best, met["MAPE"], secs)
        self.prog.configure(value=100)
        self.train_status.configure(text=f"完了：テスト MAPE {met['MAPE']:.3f}%（{secs:.1f} 秒）", fg=C["good_text"])
        self.training = False
        self.btn_train.state(["!disabled"])
        self.update_tiles()
        self._dirty = set(self.tabs)
        self._render_current()
        self.app.on_forecast_changed()

    def reset_model(self):
        if self.training:
            return
        ds = self.app.ds
        ds.pred = ds.pred_initial.copy()
        self.cur = dict(met=ds.metrics["test"], importance=self._imp_items(ds.metrics["importance_gain"]),
                        best_iter=ds.metrics["model"]["best_iter"], secs=ds.metrics["model"]["train_seconds"],
                        label="初期モデル")
        self.train_status.configure(text="初期モデルに戻しました。", fg=C["ink2"])
        self.prog.configure(value=0)
        self.update_tiles()
        self._dirty = set(self.tabs)
        self._render_current()
        self.app.on_forecast_changed()
