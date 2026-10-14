"""発電計画最適化ページ（PuLP + HiGHS）"""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np

from core.dispatch import FUEL_LABEL, GROUPS, Scenario, clone_units, default_units, evaluate_forecast_value, solve_dispatch
from ui import charts
from ui.theme import C, DISPATCH_COLOR, FONTS, px
from ui.widgets import Card, ChartCanvas, HoverTip, LabeledScale, ScrollFrame, SectionLabel, StatTile, make_tree

WEEK = "月火水木金土日"


def _dlabel(d) -> str:
    d = np.datetime64(d, "D").astype(object)
    return f"{d.year}/{d.month:02d}/{d.day:02d}（{WEEK[d.weekday()]}）"


class DispatchPage(tk.Frame):
    title = "発電計画最適化"
    subtitle = "予測需要を最小コストで満たす 24 時間の起動停止・出力配分（PuLP でモデル化、HiGHS で求解）"

    def __init__(self, parent, app):
        super().__init__(parent, bg=C["page"])
        self.app = app
        self.units = default_units()
        self.q: queue.Queue = queue.Queue()
        self.running = False
        self.result = None
        self.fv = None
        ds = app.ds
        self.days = list(ds.test_days)

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # ---------------- 左：操作パネル
        panel = Card(self, pad=0)
        panel.grid(row=0, column=0, sticky="nsw")
        sf = ScrollFrame(panel.body, width=px(292))
        sf.pack(fill="both", expand=True)
        b = tk.Frame(sf.inner, bg=C["card"])
        b.pack(fill="both", expand=True, padx=16, pady=14)
        self._build_controls(b)

        # ---------------- 右：結果
        right = tk.Frame(self, bg=C["page"])
        right.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(1, weight=1)

        tiles = tk.Frame(right, bg=C["page"])
        tiles.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        self.t_cost = StatTile(tiles, "総発電コスト（億円/日）", "-", "", small=True)
        self.t_unit = StatTile(tiles, "平均発電単価（円/kWh）", "-", "", small=True)
        self.t_co2 = StatTile(tiles, "CO2 排出量（万t/日）", "-", "", small=True)
        self.t_price = StatTile(tiles, "最高限界価格（円/kWh）", "-", "", small=True)
        self.t_solve = StatTile(tiles, "求解時間（秒）", "-", "", small=True)
        for i, t in enumerate((self.t_cost, self.t_unit, self.t_co2, self.t_price, self.t_solve)):
            t.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 10, 0))
            tiles.grid_columnconfigure(i, weight=1, uniform="t")

        card = Card(right, pad=10)
        card.grid(row=1, column=0, sticky="nsew")
        self.nb = ttk.Notebook(card.body)
        self.nb.pack(fill="both", expand=True)
        f0 = tk.Frame(self.nb, bg=C["card"])
        self.nb.add(f0, text="発電計画（24時間）")
        self.chart_title = tk.Label(f0, text="", font=FONTS["h3"], bg=C["card"], fg=C["ink"], anchor="w")
        self.chart_title.pack(fill="x", padx=4, pady=(10, 0))
        self.chart = ChartCanvas(f0, height=4.6)
        self.chart.pack(fill="both", expand=True)
        self._build_result_tabs()

        self.after(150, self._poll)

    # ------------------------------------------------------------------
    def _build_controls(self, b):
        ds = self.app.ds
        SectionLabel(b, "対象日（テスト期間）").pack(fill="x")
        row = tk.Frame(b, bg=C["card"])
        row.pack(fill="x", pady=(4, 0))
        ttk.Button(row, text="◀", width=3, style="Chip.TButton", command=lambda: self._shift_day(-1)).pack(side="left")
        self.day_var = tk.StringVar()
        self.day_cb = ttk.Combobox(row, textvariable=self.day_var, state="readonly", width=17,
                                   values=[_dlabel(d) for d in self.days])
        self.day_cb.pack(side="left", padx=4, fill="x", expand=True)
        ttk.Button(row, text="▶", width=3, style="Chip.TButton", command=lambda: self._shift_day(1)).pack(side="left")
        chips = tk.Frame(b, bg=C["card"])
        chips.pack(fill="x", pady=(6, 0))
        self.notable = ds.notable_days()
        for i, (lab, d) in enumerate(self.notable.items()):
            btn = ttk.Button(chips, text=lab, style="Chip.TButton", command=lambda d=d: self._set_day(d, run=True))
            btn.grid(row=i // 2, column=i % 2, sticky="ew", padx=(0 if i % 2 == 0 else 4, 0), pady=2)
            HoverTip(btn, _dlabel(d))
        chips.grid_columnconfigure(0, weight=1)
        chips.grid_columnconfigure(1, weight=1)

        SectionLabel(b, "需要の入力").pack(fill="x", pady=(14, 2))
        self.source = tk.StringVar(value="forecast")
        ttk.Radiobutton(b, text="LightGBM 予測値（前日計画）", value="forecast", variable=self.source).pack(anchor="w")
        ttk.Radiobutton(b, text="実績値（事後に振り返る計画）", value="actual", variable=self.source).pack(anchor="w")

        SectionLabel(b, "シナリオ").pack(fill="x", pady=(14, 2))
        self.s_lng = LabeledScale(b, "LNG 価格", 0.5, 2.0, 0.05, 1.0, "×{:.2f}")
        self.s_coal = LabeledScale(b, "石炭価格", 0.5, 2.0, 0.05, 1.0, "×{:.2f}")
        self.s_oil = LabeledScale(b, "石油価格", 0.5, 2.0, 0.05, 1.0, "×{:.2f}")
        self.s_co2 = LabeledScale(b, "CO2 価格", 0, 20000, 500, 0, "{:,.0f} 円/t")
        self.s_solar = LabeledScale(b, "太陽光の導入量（実績比）", 0.5, 3.0, 0.1, 1.0, "×{:.1f}")
        self.s_res = LabeledScale(b, "運転予備率", 0, 10, 0.5, 3.0, "{:.1f}%")
        self.s_imp = LabeledScale(b, "連系線の受電価格", 8, 25, 0.5, 14.0, "{:.1f} 円/kWh")
        tips = {self.s_lng: "基準：12 ドル/MMBtu 相当", self.s_coal: "基準：一般炭 110 ドル/t 相当",
                self.s_oil: "基準：75 ドル/バレル 相当", self.s_co2: "炭素価格。排出原単位 × 価格を限界費用に加算",
                self.s_solar: "でんき予報の太陽光発電実績に掛ける倍率（導入拡大シナリオ）",
                self.s_res: "運転中火力の余力＋揚水の発電余力を需要の何％確保するか",
                self.s_imp: "他エリアからの受電（上限 400 万kW）の単価"}
        for s in (self.s_lng, self.s_coal, self.s_oil, self.s_co2, self.s_solar, self.s_res, self.s_imp):
            s.pack(fill="x", pady=(6, 0))
            HoverTip(s, tips[s])
        self.nuc = tk.BooleanVar(value=False)
        ttk.Checkbutton(b, text="原子力 1 基（135 万kW）を運転する", variable=self.nuc).pack(anchor="w", pady=(10, 0))
        self.do_eval = tk.BooleanVar(value=True)
        cb = ttk.Checkbutton(b, text="予測誤差によるコスト増も評価する", variable=self.do_eval)
        cb.pack(anchor="w", pady=(4, 0))
        HoverTip(cb, "予測需要で決めた起動停止のまま実績需要に対応した場合のコストと、"
                     "実績需要を事前に知っていた場合の最適コストを比べます（MILP を追加で 2 回解きます）。")

        self.btn_run = ttk.Button(b, text="最適化を実行", style="Accent.TButton", command=self.run)
        self.btn_run.pack(fill="x", pady=(16, 4))
        ttk.Button(b, text="シナリオを初期値に戻す", command=self.reset).pack(fill="x")
        self.prog = ttk.Progressbar(b, mode="indeterminate")
        self.prog.pack(fill="x", pady=(10, 0))
        self.status = tk.Label(b, text="", font=FONTS["small"], bg=C["card"], fg=C["muted"], justify="left",
                               wraplength=px(260), anchor="w")
        self.status.pack(fill="x", pady=(4, 0))

        first = self.notable.get("夏の最大需要日", self.days[-1])
        self._set_day(first, run=False)
        self.day_cb.bind("<<ComboboxSelected>>", lambda e: self.run())

    def _build_result_tabs(self):
        # 電源別
        f1 = tk.Frame(self.nb, bg=C["card"])
        self.nb.add(f1, text="電源別の運転結果")
        cols = [("name", "電源"), ("fuel", "燃料"), ("cap", "定格 万kW"), ("mc", "限界費用 円/kWh"),
                ("gen", "発電量 万kWh"), ("cf", "利用率"), ("hrs", "運転 h"), ("st", "起動"), ("cost", "費用 万円")]
        fr, self.tree = make_tree(f1, cols, [150, 50, 70, 100, 90, 60, 55, 45, 80], anchors={"name": "w", "fuel": "center"}, height=7)
        fr.pack(fill="both", expand=True, pady=(8, 0))

        # 予測誤差の影響
        f2 = tk.Frame(self.nb, bg=C["card"])
        self.nb.add(f2, text="予測誤差の影響")
        tl = tk.Frame(f2, bg=C["card"])
        tl.pack(fill="x", pady=(10, 6))
        self.fv_plan = StatTile(tl, "① 前日計画（予測需要）", "-", "")
        self.fv_rt = StatTile(tl, "② 当日の実際のコスト", "-", "")
        self.fv_pf = StatTile(tl, "③ 完全予見（実績需要が既知）", "-", "")
        self.fv_extra = StatTile(tl, "予測誤差によるコスト増", "-", "")
        for i, t in enumerate((self.fv_plan, self.fv_rt, self.fv_pf, self.fv_extra)):
            t.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 8, 0))
            tl.grid_columnconfigure(i, weight=1, uniform="f")
        note = tk.Label(f2, text="② は、前日計画で決めた火力の起動停止を変えずに（緊急設置 GT・連系線・揚水は当日も調整可）、"
                                 "実績需要に合わせて出力を再配分した費用です。② − ③ が「予測が外れたことによる追加費用」で、"
                                 "需要予測の精度がそのまま経済価値になることを示します。",
                        font=FONTS["small"], bg=C["card"], fg=C["ink2"], justify="left", anchor="w")
        note.pack(fill="x")
        f2.bind("<Configure>", lambda e: note.configure(wraplength=max(e.width - 12, 200)), add="+")
        self.fv_chart = ChartCanvas(f2, height=2.4)
        self.fv_chart.pack(fill="both", expand=True, pady=(6, 0))

        # 費用の内訳
        f3 = tk.Frame(self.nb, bg=C["card"])
        self.nb.add(f3, text="費用の内訳")
        self.cost_chart = ChartCanvas(f3, height=2.2)
        self.cost_chart.pack(fill="both", expand=True, pady=(6, 0))

        # パラメータ
        f4 = tk.Frame(self.nb, bg=C["card"])
        self.nb.add(f4, text="電源パラメータ（編集可）")
        cols = [("on", "運転"), ("name", "電源"), ("fuel", "燃料"), ("cap", "定格 万kW"), ("pmin", "最低出力"),
                ("eff", "効率 %"), ("mc", "限界費用 円/kWh"), ("co2", "CO2 kg/kWh"), ("su", "起動費 万円"), ("ud", "最小運転/停止 h")]
        fr, self.ptree = make_tree(f4, cols, [44, 150, 50, 70, 64, 56, 100, 80, 76, 96], anchors={"name": "w", "on": "center", "fuel": "center"}, height=7)
        fr.pack(fill="both", expand=True, pady=(8, 0))
        bar = tk.Frame(f4, bg=C["card"])
        bar.pack(fill="x", pady=(6, 0))
        ttk.Button(bar, text="電源構成を初期値に戻す", style="Chip.TButton", command=self._reset_units).pack(side="right")
        pn = tk.Label(bar, text="行をダブルクリックすると定格・効率・起動費の変更や停止（補修中など）を設定できます。"
                                "値は東京エリアの構成を大まかに模した仮想データです。",
                      font=FONTS["small"], bg=C["card"], fg=C["ink2"], justify="left", anchor="w")
        pn.pack(side="left", fill="x", expand=True)
        bar.bind("<Configure>", lambda e: pn.configure(wraplength=max(e.width - 190, 200)), add="+")
        self.ptree.bind("<Double-1>", self._edit_unit)
        self._fill_param_tree()

    # ------------------------------------------------------------------
    def _set_day(self, d, run=False):
        d = np.datetime64(d, "D")
        i = int(np.searchsorted(np.array(self.days, dtype="datetime64[D]"), d))
        i = int(np.clip(i, 0, len(self.days) - 1))
        self.day_cb.current(i)
        if run:
            self.run()

    def _shift_day(self, k):
        i = int(np.clip(self.day_cb.current() + k, 0, len(self.days) - 1))
        self.day_cb.current(i)
        self.run()

    def scenario(self) -> Scenario:
        return Scenario(lng_mult=self.s_lng.get(), coal_mult=self.s_coal.get(), oil_mult=self.s_oil.get(),
                        co2_price=self.s_co2.get(), solar_mult=self.s_solar.get(), reserve_pct=self.s_res.get(),
                        import_price=self.s_imp.get(), nuclear_on=self.nuc.get())

    def reset(self):
        for s, v in ((self.s_lng, 1.0), (self.s_coal, 1.0), (self.s_oil, 1.0), (self.s_co2, 0), (self.s_solar, 1.0),
                     (self.s_res, 3.0), (self.s_imp, 14.0)):
            s.set(v)
        self.nuc.set(False)
        self.source.set("forecast")
        self.run()

    def _reset_units(self):
        self.units = default_units()
        self._fill_param_tree()

    def on_show(self):
        if self.result is None and not self.running:
            self.run()

    def on_forecast_changed(self):
        if self.result is not None:
            self.status.configure(text="需要予測モデルが更新されました。「最適化を実行」で新しい予測値を使って再計算します。",
                                  fg=C["accent"])

    # ------------------------------------------------------------------
    def run(self):
        if self.running:
            return
        ds = self.app.ds
        d = self.days[self.day_cb.current()]
        act, fc, sol, _ = ds.day_series(d)
        if len(act) != 24:
            self.status.configure(text="この日は 24 時間分のデータがありません。", fg=C["critical"])
            return
        sc = self.scenario()
        units = clone_units(self.units)
        src = self.source.get()
        do_eval = self.do_eval.get()
        self.running = True
        self.btn_run.state(["disabled"])
        self.prog.start(12)
        self.status.configure(text="HiGHS で求解中…（混合整数計画）", fg=C["ink2"])
        args = (d, src, act.copy(), fc.copy(), sol.copy(), sc, units, do_eval)
        threading.Thread(target=self._worker, args=args, daemon=True).start()

    def _worker(self, d, src, act, fc, sol, sc, units, do_eval):
        try:
            fv = None
            if do_eval:
                fv = evaluate_forecast_value(fc, act, sol, units, sc)
                main = fv.plan if src == "forecast" else solve_dispatch(act, sol, units, sc)
            else:
                main = solve_dispatch(fc if src == "forecast" else act, sol, units, sc)
            self.q.put(("done", d, src, main, fv, sc, units))
        except Exception as e:  # noqa: BLE001
            import traceback
            self.q.put(("error", f"{e}\n{traceback.format_exc(limit=4)}"))

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self.running = False
                self.prog.stop()
                self.btn_run.state(["!disabled"])
                if msg[0] == "done":
                    self._show(*msg[1:])
                else:
                    self.status.configure(text="最適化に失敗しました：" + msg[1][:400], fg=C["critical"])
        except queue.Empty:
            pass
        self.after(120, self._poll)

    # ------------------------------------------------------------------
    def _show(self, d, src, res, fv, sc, units):
        self.result, self.fv = res, fv
        if res.status != "Optimal" or not res.groups:
            self.status.configure(text=f"最適解が得られませんでした（{res.status}）。", fg=C["critical"])
            return
        dem_sum = res.demand.sum()                       # 万kWh
        gen_cost = sum(res.cost_by_group.values()) + res.startup_cost   # 万円
        self.t_cost.set(f"{gen_cost / 1e4:,.1f}", f"需要 {dem_sum / 1e4:,.2f} 億kWh ／ 起動費 {res.startup_cost:,.0f} 万円")
        self.t_unit.set(f"{gen_cost / dem_sum:.2f}", "燃料費・O&M・CO2 費用・受電費・起動費の合計 ÷ 需要")
        inten = res.co2_t / (dem_sum * 10)
        self.t_co2.set(f"{res.co2_t / 1e4:,.1f}", f"排出原単位 {inten:.3f} kg-CO2/kWh")
        if res.price is not None:
            hp = int(np.argmax(res.price))
            self.t_price.set(f"{res.price.max():.1f}", f"{hp}時台 ／ 日平均 {res.price.mean():.1f} 円/kWh")
        else:
            self.t_price.set("-", "")
        curt = res.curtail.sum()
        uns = res.unserved.sum()
        extra = []
        if curt > 1:
            extra.append(f"太陽光の出力制御 {curt:,.0f} 万kWh")
        if uns > 1:
            extra.append(f"供給力不足 {uns:,.0f} 万kWh")
        self.t_solve.set(f"{res.solve_seconds:.2f}", "HiGHS で最適解（MIP ギャップ 0.2% 以内）" + ("／" + "、".join(extra) if extra else ""),
                         sub_color=C["critical"] if uns > 1 else None)
        srcj = "LightGBM 予測値" if src == "forecast" else "実績値"
        self.status.configure(text=f"{_dlabel(d)}・需要={srcj}：最適化が完了しました。", fg=C["good_text"])

        # グラフ
        ch = self.chart
        ch.clear()
        gs = ch.fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0])
        ax = ch.fig.add_subplot(gs[0])
        ax2 = ch.fig.add_subplot(gs[1], sharex=ax)
        charts.plot_dispatch(ax, ax2, res)
        self.chart_title.configure(text=f"{_dlabel(d)} の発電計画（需要：{srcj}）")
        xs = np.arange(24) + 0.5

        def rows(i):
            out = [("需要", f"{res.demand[i]:,.0f} 万kW", None)]
            for k, lab in GROUPS[::-1]:
                v = res.groups[k][i]
                if v > 0.5:
                    out.append((lab, f"{v:,.0f}", None))
            if res.ps_pump[i] > 0.5:
                out.append(("揚水動力", f"−{res.ps_pump[i]:,.0f}", None))
            if res.curtail[i] > 0.5:
                out.append(("出力制御", f"{res.curtail[i]:,.0f}", None))
            if res.price is not None:
                out.append(("限界価格", f"{res.price[i]:.1f} 円/kWh", None))
            out.append(("揚水貯水量", f"{res.soc[i]:,.0f} 万kWh", None))
            return out

        ch.add_hover(ax, xs, rows, lambda i: f"{i}:00〜{i + 1}:00", vline=False, span_axes=[ax, ax2])
        ch.draw()

        # 電源別の表
        tr = self.tree
        tr.delete(*tr.get_children())
        umap = {u.key: u for u in units}
        n = 0
        for g in units:
            if not g.enabled or g.key not in res.output:
                continue
            out = res.output[g.key]
            e = out.sum()
            n += 1
            tr.insert("", "end", values=(g.name, FUEL_LABEL[g.fuel], f"{g.pmax:,.0f}", f"{g.var_cost(sc):.2f}",
                                         f"{e:,.0f}", f"{e / (g.pmax * 24) * 100:.0f}%", int(res.on[g.key].sum()),
                                         res.starts[g.key], f"{g.var_cost(sc) * e + g.startup * res.starts[g.key]:,.0f}"),
                      tags=("odd",) if n % 2 else ())
        extra_rows = [
            ("揚水発電（発電）", "-", f"{sc.ps_gen_cap:,.0f}", "-", f"{res.ps_gen.sum():,.0f}", "", int((res.ps_gen > 0.5).sum()), "", ""),
            ("揚水発電（揚水動力）", "-", f"{sc.ps_pump_cap:,.0f}", "-", f"−{res.ps_pump.sum():,.0f}", "", int((res.ps_pump > 0.5).sum()), "", ""),
            ("連系線受電", "-", f"{sc.import_cap:,.0f}", f"{sc.import_price:.2f}", f"{res.imports.sum():,.0f}", "", "", "",
             f"{sc.import_price * res.imports.sum():,.0f}"),
            ("太陽光（利用）", "-", "", "0.00", f"{res.solar_used.sum():,.0f}", "", "", "", ""),
            ("太陽光（出力制御）", "-", "", "", f"{res.curtail.sum():,.0f}", "", "", "", ""),
            ("一般水力", "-", f"{sc.hydro:,.0f}", "0.00", f"{res.hydro.sum():,.0f}", "", "", "", ""),
        ]
        if sc.nuclear_on:
            extra_rows.append(("原子力", "原子力", f"{sc.nuclear_cap:,.0f}", f"{sc.nuclear_cost:.2f}",
                               f"{res.nuclear.sum():,.0f}", "100%", 24, 0, f"{sc.nuclear_cost * res.nuclear.sum():,.0f}"))
        for r in extra_rows:
            n += 1
            tr.insert("", "end", values=r, tags=("odd",) if n % 2 else ())

        # 予測誤差の影響
        if fv is not None:
            pc = fv.plan.objective
            rc = fv.realtime.objective
            pfc = fv.perfect.objective
            ae = np.abs(fv.plan.demand - fv.perfect.demand) / fv.perfect.demand * 100
            self.fv_plan.set(f"{pc / 1e4:,.2f} 億円", f"この日の予測 MAPE {ae.mean():.2f}%")
            self.fv_rt.set(f"{rc / 1e4:,.2f} 億円", "起動停止は前日計画のまま再配分")
            self.fv_pf.set(f"{pfc / 1e4:,.2f} 億円", "実績需要で起動停止から最適化")
            ex = fv.extra_cost
            self.fv_extra.set(f"{ex:,.0f} 万円", f"完全予見比 +{ex / pfc * 100:.2f}%"
                              + ("（供給力不足を含む）" if fv.realtime.unserved is not None and fv.realtime.unserved.sum() > 1 else ""))
            fc_ = self.fv_chart
            fc_.clear()
            axf = fc_.fig.add_subplot(111)
            charts.plot_day_forecast(axf, fv.perfect.demand, fv.plan.demand)
            act_, pr_ = fv.perfect.demand, fv.plan.demand
            fc_.add_hover(axf, np.arange(24) + 0.5,
                          lambda i: [("実績", f"{act_[i]:,.0f} 万kW", None), ("予測", f"{pr_[i]:,.0f} 万kW", None),
                                     ("誤差", f"{(pr_[i] - act_[i]) / act_[i] * 100:+.1f}%", None)],
                          lambda i: f"{i}:00〜{i + 1}:00", vline=False)
            fc_.draw()
        else:
            for t in (self.fv_plan, self.fv_rt, self.fv_pf, self.fv_extra):
                t.set("-", "評価オフ")
            self.fv_chart.clear()
            self.fv_chart.draw()

        # 費用の内訳
        cc = self.cost_chart
        cc.clear()
        axc = cc.fig.add_subplot(111)
        charts.plot_cost_breakdown(axc, res)
        cc.draw()
        self._fill_param_tree(sc)

    # ------------------------------------------------------------------
    def _fill_param_tree(self, sc=None):
        sc = sc or self.scenario()
        t = self.ptree
        t.delete(*t.get_children())
        for i, g in enumerate(self.units):
            t.insert("", "end", iid=g.key, values=("●" if g.enabled else "停止", g.name, FUEL_LABEL[g.fuel], f"{g.pmax:,.0f}",
                                                   f"{g.pmin:,.0f}", f"{g.eff * 100:.1f}", f"{g.var_cost(sc):.2f}",
                                                   f"{g.co2():.3f}", f"{g.startup:,.0f}", f"{g.min_up}/{g.min_down}"),
                     tags=("odd",) if i % 2 else ())

    def _edit_unit(self, _e=None):
        sel = self.ptree.selection()
        if not sel:
            return
        g = next(u for u in self.units if u.key == sel[0])
        win = tk.Toplevel(self)
        win.title(f"{g.name} の設定")
        win.configure(bg=C["card"])
        win.transient(self.winfo_toplevel())
        win.resizable(False, False)
        body = tk.Frame(win, bg=C["card"])
        body.pack(padx=18, pady=16)
        en = tk.BooleanVar(value=g.enabled)
        ttk.Checkbutton(body, text="運転可能（チェックを外すと補修停止の扱い）", variable=en).grid(row=0, column=0, columnspan=2, sticky="w")
        vars_ = {}
        for r, (lab, key, val) in enumerate([("定格出力（万kW）", "pmax", g.pmax), ("発電効率（%）", "eff", g.eff * 100),
                                             ("起動費（万円/回）", "startup", g.startup)], start=1):
            tk.Label(body, text=lab, font=FONTS["small"], bg=C["card"], fg=C["ink2"]).grid(row=r, column=0, sticky="w", pady=4)
            v = tk.StringVar(value=f"{val:g}")
            ttk.Entry(body, textvariable=v, width=10).grid(row=r, column=1, sticky="e", pady=4, padx=(12, 0))
            vars_[key] = v
        msg = tk.Label(body, text="", font=FONTS["small"], bg=C["card"], fg=C["critical"])
        msg.grid(row=4, column=0, columnspan=2, sticky="w")

        def ok():
            try:
                pmax = float(vars_["pmax"].get())
                eff = float(vars_["eff"].get()) / 100
                su = float(vars_["startup"].get())
                if not (0 < pmax <= 2000 and 0.1 <= eff <= 0.7 and 0 <= su <= 100000):
                    raise ValueError
            except ValueError:
                msg.configure(text="定格 0〜2,000 万kW、効率 10〜70%、起動費 0 以上で入力してください。")
                return
            ratio = g.pmin / g.pmax if g.pmax else 0.3
            rr = g.ramp / g.pmax if g.pmax else 0.5
            g.pmax, g.eff, g.startup, g.enabled = pmax, eff, su, en.get()
            g.pmin, g.ramp = round(pmax * ratio, 1), round(pmax * rr, 1)
            self._fill_param_tree()
            win.destroy()

        bar = tk.Frame(body, bg=C["card"])
        bar.grid(row=5, column=0, columnspan=2, sticky="e", pady=(10, 0))
        ttk.Button(bar, text="キャンセル", command=win.destroy).pack(side="right")
        ttk.Button(bar, text="適用", style="Accent.TButton", command=ok).pack(side="right", padx=(0, 8))
        win.grab_set()
