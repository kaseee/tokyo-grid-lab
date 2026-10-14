"""データと手法ページ"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from ui.theme import C, FONTS
from ui.widgets import Card


def _content(ds):
    met = ds.metrics
    p = met["periods"]
    m = met["model"]
    return [
        ("h", "データの出典"),
        ("b", "■ 電力需要・太陽光発電実績"),
        ("p", "東京電力パワーグリッド「でんき予報」過去の電力使用実績データ（2022年4月〜2026年10月、月別 zip に含まれる日別 CSV）。\n"
              "1時間値の「当日実績(万kW)」を目的変数、5分値の「太陽光発電実績」を時間平均して発電計画の太陽光出力に使用。\n"
              "https://www.tepco.co.jp/forecast/html/download-j.html"),
        ("b", "■ 気象"),
        ("p", "気象庁「過去の気象データ・ダウンロード」時別値：東京・横浜・千葉・熊谷・水戸・宇都宮・前橋・甲府の気温、相対湿度、日照時間。\n"
              "各都県の人口（2020年国勢調査）で重み付けした「関東平均」を作成。\n"
              "https://www.data.jma.go.jp/risk/obsdl/index.php"),
        ("s", "出典：東京電力パワーグリッド株式会社「でんき予報」、気象庁ホームページ。いずれも本アプリで加工して使用しています。"),

        ("h", "需要予測（LightGBM）"),
        ("p", f"・予測の想定：対象日の前日午前に翌日 24 時間分を予測する「前日予測」。需要のラグ特徴量は 2 日前・7 日前など前日午前に確定している値だけを使用。\n"
              f"・気象は実測値を「完全な気象予報」の代わりに使用（実運用では予報誤差の分だけ精度が下がる点に注意）。\n"
              f"・特徴量（32 個）：カレンダー 13（時刻・曜日・祝日・お盆・年末年始・GW・飛び石平日 など）、気象 13（気温・湿度・日照・不快指数・日最高/最低気温・冷暖房度時 など）、需要ラグ 6。\n"
              f"・期間：学習 {p['train_start'][:10]}〜{p['train_end'][:10]}（{p['n_train']:,} 時間）、"
              f"うち {p['fit_end']} 以降を early stopping 用の検証に使い、決めた本数×1.1 で学習期間全体を再学習。"
              f"テスト {p['test_start'][:10]}〜{p['test_end'][:10]}（{p['n_test']:,} 時間）は学習に一切使っていません。\n"
              f"・モデル：LightGBM（学習率 {m['params']['learning_rate']}、num_leaves {m['params']['num_leaves']}、"
              f"min_data_in_leaf {m['params']['min_data_in_leaf']}、決定木 {m['best_iter']:,} 本）。log(需要) を二乗誤差で学習。\n"
              f"・評価指標：MAPE = 平均( |予測 − 実績| ÷ 実績 ) × 100。テスト期間の MAPE は {met['test']['MAPE']:.3f}%、"
              f"前週同時刻の値をそのまま使う方法は {met['baselines']['naive_week']['MAPE']:.3f}%。"),

        ("h", "発電計画の最適化（PuLP + HiGHS）"),
        ("p", "・対象：1 日 24 時間。需要（LightGBM 予測値 または 実績値）を最小コストで満たす各電源の出力と火力の起動停止を決める混合整数計画（MILP）。\n"
              "・目的関数：Σ（限界費用 × 出力）＋ 起動費 ＋ 連系線受電費 ＋ 太陽光出力制御・供給力不足のペナルティ。\n"
              "　限界費用 = 燃料単価 ÷ 発電効率 ＋ O&M ＋ CO2 価格 × 排出原単位。\n"
              "・制約：需給バランス、最低出力・定格出力（運転中のみ）、出力変化速度、最小運転・停止時間、"
              "運転予備率（運転中火力の余力＋揚水の発電余力 ≥ 需要 × 予備率）、揚水の貯水量収支（効率 70%、1 日で元に戻る周期条件）、"
              "揚水の発電・揚水動力の同時運転禁止（0/1 変数）。\n"
              "・限界価格：MILP の解で起動停止を固定した LP を解き直し、需給バランス制約の双対値（円/kWh）として算出。\n"
              "・予測誤差の評価：予測需要で決めた起動停止を固定したまま実績需要に対応した費用と、実績需要で最適化した費用の差。"),
        ("b", "■ 電源構成と前提（仮想データ）"),
        ("p", "石炭 6 基（540 万kW）、LNG 複合 11 基（2,550 万kW）、LNG 汽力 3 基（690 万kW）、石油 2 基（350 万kW）、緊急設置ガスタービン 150 万kW、"
              "揚水 900 万kW（上池 6,000 万kWh）、一般水力 150 万kW、連系線受電 400 万kW。\n"
              "燃料価格は LNG 12 ドル/MMBtu、一般炭 110 ドル/t、原油 75 ドル/バレル、為替 150 円/ドル程度を想定した概数。"
              "CO2 排出係数は燃料別の炭素排出係数から換算。いずれも実在の発電所・事業者のデータではなく、手法を示すための仮定値です。"),

        ("h", "使用ソフトウェア"),
        ("p", "Python 3.12 / LightGBM 4.7（MIT）/ PuLP 3.3（BSD）/ HiGHS 1.13（MIT, highspy）/ NumPy / SciPy / matplotlib / Tkinter。\n"
              "フォント：Noto Sans CJK JP（SIL Open Font License 1.1、使用文字のみ同梱）。"),
        ("s", "このアプリは公開データを用いた個人のポートフォリオ作品であり、東京電力グループ・気象庁とは関係ありません。"),
    ]


class AboutPage(tk.Frame):
    title = "データと手法"
    subtitle = "出典・前提・定式化"

    def __init__(self, parent, app):
        super().__init__(parent, bg=C["page"])
        card = Card(self, pad=18)
        card.pack(fill="both", expand=True)
        txt = tk.Text(card.body, wrap="word", bg=C["card"], fg=C["ink"], relief="flat", bd=0, padx=4, pady=2,
                      highlightthickness=0, cursor="arrow", spacing1=2, spacing3=2)
        vsb = ttk.Scrollbar(card.body, orient="vertical", command=txt.yview)
        txt.configure(yscrollcommand=vsb.set)
        txt.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")
        txt.tag_configure("h", font=FONTS["h2"], foreground=C["ink"], spacing1=16, spacing3=6)
        txt.tag_configure("b", font=FONTS["body_b"], foreground=C["ink"], spacing1=8, spacing3=2)
        txt.tag_configure("p", font=FONTS["body"], foreground=C["ink2"], lmargin1=0, lmargin2=14, spacing2=4)
        txt.tag_configure("s", font=FONTS["small"], foreground=C["muted"], spacing1=6)
        first = True
        for kind, text in _content(app.ds):
            if kind == "h" and first:
                txt.insert("end", text + "\n", ("h",))
                txt.tag_configure("h1st", spacing1=0)
                first = False
                continue
            txt.insert("end", text + "\n", (kind,))
        txt.configure(state="disabled")
