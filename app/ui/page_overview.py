"""概要ページ"""
from __future__ import annotations

import tkinter as tk

import numpy as np
from matplotlib import dates as mdates

from ui import charts
from ui.theme import C, FONTS
from ui.widgets import Card, ChartCanvas, StatTile

WEEK = "月火水木金土日"


def _d(s: str) -> str:
    y, m, d = s[:10].split("-")
    return f"{int(y)}/{int(m)}/{int(d)}"


class OverviewPage(tk.Frame):
    title = "概要"
    subtitle = "公開データだけで作る「需要予測 → 発電計画」の一気通貫デモ"

    def __init__(self, parent, app):
        super().__init__(parent, bg=C["page"])
        self.app = app
        ds = app.ds
        met = ds.metrics
        p = met["periods"]

        tiles = tk.Frame(self, bg=C["page"])
        tiles.pack(fill="x", pady=(0, 12))
        naive = met["baselines"]["naive_week"]["MAPE"]
        test = met["test"]
        red = (1 - test["MAPE"] / naive) * 100
        self.t_mape = StatTile(tiles, "需要予測の誤差率 MAPE（テスト期間）", f"{test['MAPE']:.2f}%",
                               f"前週同時刻をそのまま使う方法（{naive:.2f}%）より誤差を {red:.0f}% 削減")
        self.t_peak = StatTile(tiles, "日最大需要の誤差率（ピーク MAPE）", f"{test['peak_MAPE']:.2f}%",
                               f"平均絶対誤差 {test['MAE']:.0f} 万kW ／ 決定係数 R² {test['R2']:.3f}")
        self.t_data = StatTile(tiles, "学習・テストに使った時間数", f"{p['n_train'] + p['n_test']:,} 時間",
                               f"学習 {_d(p['train_start'])}〜{_d(p['train_end'])} ／ テスト {_d(p['test_start'])}〜{_d(p['test_end'])}")
        self.t_opt = StatTile(tiles, "発電計画モデル（1日・24時間）", "23 火力 + 揚水",
                              "起動停止つき MILP を PuLP で定式化し HiGHS で求解")
        for i, t in enumerate((self.t_mape, self.t_peak, self.t_data, self.t_opt)):
            t.grid(row=0, column=i, sticky="nsew", padx=(0 if i == 0 else 10, 0))
            tiles.grid_columnconfigure(i, weight=1, uniform="t")

        bottom = tk.Frame(self, bg=C["page"])
        bottom.pack(side="bottom", fill="x", pady=(12, 0))
        card = Card(self, "東京エリアの電力需要（日別）", "1時間値を日ごとに集計。背景色は学習・検証・テストの期間区分")
        card.pack(fill="both", expand=True)
        self.chart = ChartCanvas(card.body, height=3.0)
        self.chart.pack(fill="both", expand=True)

        flow = Card(bottom, "処理の流れ")
        flow.grid(row=0, column=0, sticky="nsew")
        self._flow(flow.body)
        abl = Card(bottom, "何が予測精度に効いているか", "特徴量グループ別のテスト MAPE")
        abl.grid(row=0, column=1, sticky="nsew", padx=(12, 0))
        bottom.grid_columnconfigure(0, weight=11, uniform="b")
        bottom.grid_columnconfigure(1, weight=9, uniform="b")
        self.abl_chart = ChartCanvas(abl.body, height=2.0)
        self.abl_chart.pack(fill="both", expand=True)
        self.render()

    def _flow(self, body):
        met = self.app.ds.metrics
        steps = [
            ("1", "データ取得", "でんき予報（需要・太陽光）と気象庁の関東8地点の気象"),
            ("2", "特徴量", "祝日・お盆、人口加重気温、2日前・7日前の需要など32個"),
            ("3", "LightGBM", f"前日予測を学習し、テスト MAPE {met['test']['MAPE']:.2f}%"),
            ("4", "最適化", "PuLP で MILP を定式化し HiGHS で最小コスト計画"),
        ]
        row = tk.Frame(body, bg=C["card"])
        row.pack(fill="both", expand=True)
        for i, (num, head, desc) in enumerate(steps):
            box = tk.Frame(row, bg="#f7f7f4", highlightthickness=1, highlightbackground=C["border"])
            box.grid(row=0, column=i * 2, sticky="nsew")
            top = tk.Frame(box, bg="#f7f7f4")
            top.pack(fill="x", padx=10, pady=(9, 2))
            tk.Label(top, text=num, font=FONTS["small_b"], bg=C["accent"], fg="#ffffff", width=2).pack(side="left")
            tk.Label(top, text=head, font=FONTS["body_b"], bg="#f7f7f4", fg=C["ink"]).pack(side="left", padx=(6, 0))
            lb = tk.Label(box, text=desc, font=FONTS["small"], bg="#f7f7f4", fg=C["ink2"], justify="left",
                          anchor="nw")
            lb.pack(fill="both", expand=True, padx=10, pady=(0, 10))
            box.bind("<Configure>", lambda e, lb=lb: lb.configure(wraplength=max(e.width - 22, 60)))
            row.grid_columnconfigure(i * 2, weight=1, uniform="f")
            if i < len(steps) - 1:
                tk.Label(row, text="→", font=FONTS["h2"], bg=C["card"], fg=C["muted"]).grid(row=0, column=i * 2 + 1,
                                                                                         padx=4)

    def render(self):
        ds = self.app.ds
        ch = self.chart
        ch.clear()
        ax = ch.fig.add_subplot(111)
        x, mean, mn, mx, days = charts.plot_overview(ax, ds)

        def rows(i):
            return [("日平均", f"{mean[i]:,.0f} 万kW", None), ("日最大", f"{mx[i]:,.0f} 万kW", None),
                    ("日最小", f"{mn[i]:,.0f} 万kW", None)]

        def title(i):
            d = days[i].astype(object)
            return f"{d.year}/{d.month}/{d.day}（{WEEK[d.weekday()]}）"

        ch.add_hover(ax, x, rows, title)
        ch.draw()

        met = ds.metrics
        lab = {"calendar": "カレンダーのみ", "calendar+lag": "カレンダー＋需要ラグ",
               "calendar+weather": "カレンダー＋気象", "all": "全特徴量（本モデル）"}
        rows_ = [("前週同時刻（比較用）", met["baselines"]["naive_week"]["MAPE"], False)]
        rows_ += [(lab[a["key"]], a["MAPE"], a["key"] == "all") for a in met["ablation"]]
        ac = self.abl_chart
        ac.clear()
        ax2 = ac.fig.add_subplot(111)
        charts.plot_ablation(ax2, rows_)
        ax2.set_title("")
        ac.draw()
