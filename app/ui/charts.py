"""グラフ描画関数（matplotlib）。各関数は与えられた Axes に描くだけで、ホバー登録は呼び出し側で行う。"""
from __future__ import annotations

import datetime as dt

import numpy as np
from matplotlib import dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator

from core.dispatch import GROUPS
from ui.theme import C, DAYTYPE_COLOR, DAYTYPE_LABEL, DISPATCH_COLOR, GROUP_COLOR, NEG_ERR, POS_ERR, SERIES

THOUSANDS = FuncFormatter(lambda x, _: f"{x:,.0f}")


def to_num(ts) -> np.ndarray:
    """numpy datetime64 → matplotlib の日付数値"""
    return mdates.date2num(np.asarray(ts).astype("datetime64[s]").astype(object))


def md_formatter(with_year=False):
    def f(x, _):
        d = mdates.num2date(x)
        return f"{d.year}/{d.month}" if with_year else f"{d.month}/{d.day}"
    return FuncFormatter(f)


def finish_axes(ax, ylabel=None, yfmt=THOUSANDS):
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(C["axis"])
    if yfmt is not None:
        ax.yaxis.set_major_formatter(yfmt)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5, color=C["muted"], labelpad=6)
    ax.tick_params(axis="y", length=0, pad=4)
    ax.margins(x=0.01)


def cap_bars(ax, bars, max_px=22, orient="h"):
    """棒の太さを画面上で max_px 以下に保つ（図の大きさが変わっても細身のまま）"""
    bars = list(bars)
    if not bars:
        return
    base = [b.get_height() if orient == "h" else b.get_width() for b in bars]
    cen = [(b.get_y() + b.get_height() / 2) if orient == "h" else (b.get_x() + b.get_width() / 2) for b in bars]

    def hook():
        tr = ax.transData
        if orient == "h":
            ppu = abs(tr.transform((0, 1))[1] - tr.transform((0, 0))[1])
        else:
            ppu = abs(tr.transform((1, 0))[0] - tr.transform((0, 0))[0])
        if ppu <= 0:
            return False
        changed = False
        for b, b0, c in zip(bars, base, cen):
            t = min(b0, max_px / ppu)
            cur = b.get_height() if orient == "h" else b.get_width()
            if abs(cur - t) > 1e-4:
                changed = True
                if orient == "h":
                    b.set_height(t)
                    b.set_y(c - t / 2)
                else:
                    b.set_width(t)
                    b.set_x(c - t / 2)
        return changed

    hooks = getattr(ax.figure, "_draw_hooks", None)
    if hooks is not None:
        hooks.append(hook)


def legend_top(ax, handles=None, labels=None, ncol=4, y=1.0):
    kw = dict(loc="lower left", bbox_to_anchor=(0, y), ncol=ncol, handlelength=1.4, handleheight=0.8,
              columnspacing=1.4, borderaxespad=0.2, fontsize=8.5)
    if handles is not None:
        return ax.legend(handles, labels, **kw)
    return ax.legend(**kw)


# ---------------------------------------------------------------------------
# 概要：全期間の需要
# ---------------------------------------------------------------------------
def plot_overview(ax, ds):
    days, mean, mn, mx = ds.daily_profile()
    x = to_num(days)
    p = ds.metrics["periods"]
    spans = [
        (p["train_start"][:10], p["fit_end"], C["band_train"], "学習"),
        (p["fit_end"], p["test_start"][:10], C["band_valid"], "検証（early stopping）"),
        (p["test_start"][:10], str(days[-1] + np.timedelta64(1, "D")), C["band_test"], "テスト（未使用データ）"),
    ]
    for s, e, col, lab in spans:
        xs, xe = to_num(np.array([s], dtype="datetime64[D]"))[0], to_num(np.array([e[:10]], dtype="datetime64[D]"))[0]
        ax.axvspan(xs, xe, color=col, lw=0, zorder=0)
        ax.text((xs + xe) / 2, 1.0, lab, transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                fontsize=8.5, color=C["ink2"])
    ax.fill_between(x, mn, mx, color=SERIES[0], alpha=0.14, lw=0, label="日最小〜日最大")
    ax.plot(x, mean, color=SERIES[0], lw=1.3, label="日平均")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
    ax.xaxis.set_major_formatter(md_formatter(with_year=True))
    finish_axes(ax, "万kW")
    ax.set_xlim(x[0], x[-1] + 1)
    ax.set_ylim(0, max(mx) * 1.08)
    legend_top(ax, ncol=2, y=1.07)
    return x, mean, mn, mx, days


# ---------------------------------------------------------------------------
# 需要予測：時系列（実績 vs 予測）と誤差
# ---------------------------------------------------------------------------
def plot_forecast_window(ax, ax_err, ds, start, n_days, show_tepco=False):
    s = np.datetime64(start, "h")
    e = s + np.timedelta64(24 * n_days, "h")
    m = (ds.ts >= s) & (ds.ts < e)
    ts = ds.ts[m]
    x = to_num(ts)
    y = ds.demand[m]
    p = ds.pred[m]
    # 休日の背景
    dtyp = ds.daytype[m]
    days = ts.astype("datetime64[D]")
    for d in np.unique(days):
        k = days == d
        if dtyp[k][0] == 2 or dtyp[k][0] == 1:
            xs = to_num(np.array([d]))[0]
            ax.axvspan(xs, xs + 1, color="#f1f0ec" if dtyp[k][0] == 1 else "#ebeae5", lw=0, zorder=0)
            ax_err.axvspan(xs, xs + 1, color="#f1f0ec" if dtyp[k][0] == 1 else "#ebeae5", lw=0, zorder=0)
    ax.plot(x, y, color=C["ink"], lw=1.6, label="実績", zorder=3)
    ax.plot(x, p, color=SERIES[0], lw=2.0, label="LightGBM 予測（前日）", zorder=4)
    if show_tepco:
        ax.plot(x, ds.tepco[m], color=SERIES[1], lw=1.2, label="東電 当日予測値（参考）", zorder=2)
    finish_axes(ax, "万kW")
    lo = np.nanmin(np.r_[y, p]) * 0.9
    hi = np.nanmax(np.r_[y, p]) * 1.06
    ax.set_ylim(lo, hi)
    hdl, lab = ax.get_legend_handles_labels()
    hdl += [Patch(color="#ebeae5"), Patch(color=POS_ERR), Patch(color=NEG_ERR)]
    lab += ["日曜・祝日", "誤差：予測＞実績", "誤差：予測＜実績"]
    legend_top(ax, hdl, lab, ncol=6, y=1.0)

    err = (p - y) / y * 100
    cols = np.where(err >= 0, POS_ERR, NEG_ERR)
    ax_err.bar(x, err, width=1 / 24 * 0.8, color=cols, align="edge", lw=0)
    ax_err.axhline(0, color=C["axis"], lw=0.8)
    lim = max(4.0, np.nanpercentile(np.abs(err), 99) * 1.15)
    ax_err.set_ylim(-lim, lim)
    finish_axes(ax_err, "誤差 %", yfmt=FuncFormatter(lambda v, _: f"{v:+.0f}" if v else "0"))
    ax_err.yaxis.set_major_locator(MaxNLocator(4, symmetric=True))
    for a in (ax, ax_err):
        a.xaxis.set_major_locator(mdates.DayLocator(interval=1 if n_days <= 14 else 7))
        a.xaxis.set_major_formatter(md_formatter())
        a.set_xlim(x[0], x[-1] + 1 / 24)
    ax.tick_params(labelbottom=False)
    return dict(x=x, ts=ts, y=y, p=p, err=err, tepco=ds.tepco[m], temp=ds.temp[m])


# ---------------------------------------------------------------------------
# 月別・時間帯別 MAPE
# ---------------------------------------------------------------------------
def plot_monthly_mape(ax, months, vals, overall):
    x = np.arange(len(months))
    cap_bars(ax, ax.bar(x, vals, width=0.52, color=SERIES[0], lw=0), max_px=24, orient="v")
    ax.axhline(overall, color=C["ink"], lw=1.0, zorder=3)
    ax.text(-0.45, overall, f"全体 {overall:.2f}%", va="bottom", ha="left", fontsize=8.5, color=C["ink"],
            bbox=dict(boxstyle="square,pad=0.15", fc=C["card"], ec="none"), zorder=4)
    labels = []
    mo = months.astype(object)
    for k, m in enumerate(mo):
        labels.append(f"{m.month}月" + (f"\n{m.year}" if (m.month == 1 or k == 0) else "\n"))
    ax.set_xticks(x, labels)
    imax, imin = int(np.argmax(vals)), int(np.argmin(vals))
    for i in {imax, imin}:
        ax.text(i, vals[i], f"{vals[i]:.2f}", ha="center", va="bottom", fontsize=8, color=C["ink2"])
    finish_axes(ax, "MAPE %", yfmt=FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("月別 MAPE（テスト期間）")
    return x


def plot_hourly_mape(ax, vals):
    x = np.arange(24)
    cap_bars(ax, ax.bar(x, vals, width=0.6, color=SERIES[0], lw=0), max_px=18, orient="v")
    ax.set_xticks(range(0, 24, 3), [f"{h}時" for h in range(0, 24, 3)])
    i = int(np.argmax(vals))
    ax.text(i, vals[i], f"{vals[i]:.2f}", ha="center", va="bottom", fontsize=8, color=C["ink2"])
    finish_axes(ax, "MAPE %", yfmt=FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("時間帯別 MAPE（テスト期間）")
    return x


# ---------------------------------------------------------------------------
# 散布図・重要度・気温との関係・アブレーション
# ---------------------------------------------------------------------------
def plot_scatter(ax, y, p, met):
    ax.scatter(y, p, s=5, color=SERIES[0], alpha=0.22, lw=0, rasterized=True)
    lo, hi = min(y.min(), p.min()) * 0.95, max(y.max(), p.max()) * 1.03
    ax.plot([lo, hi], [lo, hi], color=C["ink"], lw=1.0)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    finish_axes(ax, "予測（万kW）")
    ax.xaxis.set_major_formatter(THOUSANDS)
    ax.set_xlabel("実績（万kW）", fontsize=8.5, color=C["muted"])
    ax.grid(True, axis="both")
    ax.text(0.03, 0.97, f"MAPE {met['MAPE']:.2f}%\nR² {met['R2']:.4f}\nn = {met['n']:,}", transform=ax.transAxes,
            va="top", ha="left", fontsize=9, color=C["ink"],
            bbox=dict(boxstyle="round,pad=0.5", fc="#ffffff", ec=C["border"], lw=1))
    ax.set_title("予測値と実績値（テスト期間・1時間値）")


def plot_importance(ax, items, top=15):
    items = items[:top][::-1]
    y = np.arange(len(items))
    vals = [it["value"] for it in items]
    cols = [GROUP_COLOR[it["group"]] for it in items]
    cap_bars(ax, ax.barh(y, vals, height=0.56, color=cols, lw=0), max_px=18)
    ax.set_yticks(y, [it["label"] for it in items])
    ax.tick_params(axis="y", labelcolor=C["ink2"], labelsize=8.5)
    for yi, v in zip(y, vals):
        ax.text(v, yi, f" {v:.1f}%", va="center", ha="left", fontsize=8, color=C["ink2"])
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlim(0, max(vals) * 1.18)
    hdl = [Patch(color=GROUP_COLOR[g]) for g in ("calendar", "weather", "lag")]
    legend_top(ax, hdl, ["カレンダー", "気象", "需要ラグ"], ncol=3, y=1.0)
    ax.set_title("特徴量の重要度（gain 比率・上位15）", pad=26)


def plot_temp_demand(ax, temp, demand, daytype, n_max=12000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(temp))
    ok = np.isfinite(temp) & np.isfinite(demand)
    idx = idx[ok]
    if len(idx) > n_max:
        idx = rng.choice(idx, n_max, replace=False)
    for k in (0, 1, 2):
        m = idx[daytype[idx] == k]
        ax.scatter(temp[m], demand[m], s=5, color=DAYTYPE_COLOR[k], alpha=0.28, lw=0, label=DAYTYPE_LABEL[k],
                   rasterized=True)
    finish_axes(ax, "需要（万kW）")
    ax.set_xlabel("気温（関東8地点の人口加重平均, ℃）", fontsize=8.5, color=C["muted"])
    ax.grid(True, axis="both")
    leg = legend_top(ax, ncol=3, y=1.0)
    for h in leg.legend_handles:
        h.set_alpha(1)
        h.set_sizes([30])
    ax.set_title("気温と電力需要（学習＋テスト期間・1時間値）", pad=24)


def plot_ablation(ax, rows):
    """rows: [(label, mape, highlight)]"""
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, hl) in zip(y, rows):
        cap_bars(ax, ax.barh(yi, v, height=0.5, color=SERIES[0] if hl else "#c9c8c1", lw=0), max_px=22)
        ax.text(v, yi, f"  {v:.2f}%", va="center", ha="left", fontsize=9, color=C["ink"] if hl else C["ink2"],
                fontweight="bold" if hl else "normal")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", labelcolor=C["ink2"], labelsize=9)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlim(0, max(r[1] for r in rows) * 1.25)
    ax.set_title("特徴量の組み合わせ別 テスト MAPE")


# ---------------------------------------------------------------------------
# 発電計画
# ---------------------------------------------------------------------------
def _step(v):
    v = np.asarray(v, float)
    return np.r_[v, v[-1]]


def plot_dispatch(ax, ax_price, res, title=""):
    T = len(res.demand)
    x = np.arange(T + 1)
    bottoms = np.zeros(T + 1)
    handles, labels = [], []
    for key, lab in GROUPS:
        vals = _step(res.groups[key])
        if vals.max() <= 0.5:
            continue
        ax.fill_between(x, bottoms, bottoms + vals, step="post", color=DISPATCH_COLOR[key], lw=1.2,
                        edgecolor=C["card"], alpha=0.92, zorder=2)
        bottoms = bottoms + vals
        handles.append(Patch(color=DISPATCH_COLOR[key]))
        labels.append(lab)
    if res.curtail is not None and res.curtail.sum() > 1:
        c = _step(res.curtail)
        ax.fill_between(x, bottoms, bottoms + c, step="post", facecolor="none", edgecolor=DISPATCH_COLOR["solar"],
                        hatch="////", lw=0, zorder=2)
        handles.append(Patch(facecolor="none", edgecolor=DISPATCH_COLOR["solar"], hatch="////"))
        labels.append("太陽光（出力制御）")
    if res.unserved is not None and res.unserved.sum() > 1:
        u = _step(res.unserved)
        ax.fill_between(x, bottoms, bottoms + u, step="post", color=C["critical"], hatch="xx", lw=0, zorder=2)
        handles.append(Patch(color=C["critical"]))
        labels.append("供給力不足")
    pump = _step(res.ps_pump)
    if pump.max() > 0.5:
        ax.fill_between(x, 0, -pump, step="post", color=DISPATCH_COLOR["pumped"], alpha=0.45, lw=0, zorder=2)
        handles.append(Patch(color=DISPATCH_COLOR["pumped"], alpha=0.45))
        labels.append("揚水動力（消費）")
    ax.step(x, _step(res.demand), where="post", color=C["ink"], lw=2.0, zorder=5)
    handles.append(Line2D([], [], color=C["ink"], lw=2))
    labels.append("需要")
    ax.axhline(0, color=C["axis"], lw=0.8, zorder=1)
    finish_axes(ax, "万kW")
    ax.set_xlim(0, T)
    top = max(bottoms.max(), res.demand.max()) * 1.04
    ax.set_ylim(-pump.max() * 1.15 if pump.max() > 0.5 else 0, top)
    ax.set_xticks(range(0, T + 1, 3), [f"{h}時" for h in range(0, T + 1, 3)])
    ax.tick_params(labelbottom=False)
    legend_top(ax, handles, labels, ncol=7, y=1.0)
    if title:
        ax.set_title(title)

    if res.price is not None:
        ax_price.step(x, _step(res.price), where="post", color=C["ink"], lw=2.0)
        pmax = float(np.max(res.price))
        imax = int(np.argmax(res.price))
        ax_price.text(imax + 0.5, pmax, f"{pmax:.1f}", ha="center", va="bottom", fontsize=8, color=C["ink2"])
        pmin = float(np.min(res.price))
        ax_price.set_ylim(min(0.0, pmin - 1.0), max(pmax * 1.3, 5))
        if pmin < 0:
            ax_price.axhline(0, color=C["axis"], lw=0.8)
    else:
        ax_price.text(0.5, 0.5, "限界価格を計算できませんでした", transform=ax_price.transAxes, ha="center",
                      color=C["muted"])
    finish_axes(ax_price, "円/kWh", yfmt=FuncFormatter(lambda v, _: f"{v:.0f}"))
    ax_price.set_xlim(0, T)
    ax_price.set_xticks(range(0, T + 1, 3), [f"{h}時" for h in range(0, T + 1, 3)])
    ax_price.set_title("限界価格（需給バランス制約の双対値, 円/kWh）", fontsize=9, pad=4)
    return x


def plot_cost_breakdown(ax, res):
    rows = [(lab, res.cost_by_group.get(k, 0.0) / 1e4, DISPATCH_COLOR[k]) for k, lab in GROUPS
            if res.cost_by_group.get(k, 0.0) > 0]
    rows.append(("起動費", res.startup_cost / 1e4, "#a9a8a1"))
    rows = [r for r in rows if r[1] > 0.005]
    y = np.arange(len(rows))[::-1]
    for yi, (lab, v, col) in zip(y, rows):
        cap_bars(ax, ax.barh(yi, v, height=0.5, color=col, lw=0), max_px=22)
        ax.text(v, yi, f"  {v:.2f}億円", va="center", ha="left", fontsize=8.5, color=C["ink2"])
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", labelcolor=C["ink2"], labelsize=8.5)
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.35 if rows else 1)
    ax.set_title("費用の内訳（燃料費・O&M・CO2費用・受電費・起動費）")


def plot_day_forecast(ax, actual, forecast):
    """発電計画ページ用：対象日の需要（予測と実績）"""
    x = np.arange(len(actual) + 1)
    ax.fill_between(x, _step(forecast), _step(actual), step="post", color=SERIES[0], alpha=0.12, lw=0,
                    label="予測と実績の差")
    ax.step(x, _step(actual), where="post", color=C["ink"], lw=1.8, label="実績")
    ax.step(x, _step(forecast), where="post", color=SERIES[0], lw=2.0, label="LightGBM 予測")
    finish_axes(ax, "万kW")
    ax.set_xlim(0, len(actual))
    lo = min(np.min(actual), np.min(forecast)) * 0.92
    hi = max(np.max(actual), np.max(forecast)) * 1.05
    ax.set_ylim(lo, hi)
    ax.set_xticks(range(0, 25, 3), [f"{h}時" for h in range(0, 25, 3)])
    legend_top(ax, ncol=3)
