"""
発電計画（起動停止＋経済負荷配分）の最適化コア
  モデラー : PuLP
  ソルバー : HiGHS（highspy 経由、無償の MILP/LP ソルバー）

1 日 24 時間を対象に、需要（LightGBM 予測値または実績値）を最小コストで満たす各電源の出力を決める。

  決定変数  p[g,t] 出力（万kW） / u[g,t] 運転状態(0/1) / v,w 起動・停止
            揚水の発電・揚水動力・貯水量・運転モード(0/1)、連系線受電、太陽光の利用量
  目的関数  燃料費＋運転維持費＋CO2 費用＋起動費＋受電費用＋出力制御・供給不足ペナルティ（単位：万円）
  主な制約  需給バランス、最小出力・最大出力、出力変化速度、最小運転・停止時間、
            運転予備力（需要×予備率）、揚水の貯水量収支（1 日周期）

単位：電力 万kW、電力量 万kWh、費用 万円（1 万kW × 1 時間 × c 円/kWh = c 万円）。
需給バランス制約の双対値はそのまま「円/kWh」の限界費用（システム限界価格）になる。

※ 電源構成・パラメータは東京エリアの電源構成を大まかに模した仮想データ（実在発電所の値ではない）。
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field

import numpy as np
import pulp

# ---------------------------------------------------------------------------
# 燃料単価（熱量当たり、円/kWh-熱）と CO2 排出係数（kg-CO2/kWh-熱）
#   LNG 12 ドル/MMBtu、一般炭 110 ドル/t（6,000kcal/kg）、原油・重油 75 ドル/bbl 相当、150 円/ドル
#   排出係数は環境省「算定・報告・公表制度」の燃料別炭素排出係数から換算
# ---------------------------------------------------------------------------
FUEL_PRICE_TH = {"coal": 2.36, "lng": 6.14, "oil": 7.00, "nuclear": 0.0, "none": 0.0}
FUEL_CO2_TH = {"coal": 0.326, "lng": 0.178, "oil": 0.257, "nuclear": 0.0, "none": 0.0}
FUEL_LABEL = {"coal": "石炭", "lng": "LNG", "oil": "石油", "nuclear": "原子力", "none": "-"}


@dataclass
class Unit:
    key: str
    name: str
    group: str          # 積み上げグラフの区分
    fuel: str
    pmax: float         # 万kW
    pmin: float         # 万kW（運転中の最低出力）
    eff: float          # 発電端効率（0-1）
    om: float           # 運転維持費など（円/kWh）
    ramp: float         # 出力変化速度（万kW/h）
    min_up: int         # 最小運転時間（h）
    min_down: int       # 最小停止時間（h）
    startup: float      # 起動費（万円/回）
    fast: bool = False  # 当日でも起動判断できる電源（GT 等）
    enabled: bool = True

    def var_cost(self, sc: "Scenario") -> float:
        """限界費用（円/kWh）＝燃料費＋O&M＋CO2 費用"""
        mult = {"coal": sc.coal_mult, "lng": sc.lng_mult, "oil": sc.oil_mult}.get(self.fuel, 1.0)
        fuel = FUEL_PRICE_TH[self.fuel] * mult / self.eff
        co2 = sc.co2_price * self.co2() / 1000.0
        return fuel + self.om + co2

    def co2(self) -> float:
        """kg-CO2/kWh"""
        return FUEL_CO2_TH[self.fuel] / self.eff


def _units(prefix, n, name, group, fuel, pmax, pmin_ratio, eff0, eff_step, om, ramp_ratio, ut, dt, su_per, fast=False):
    out = []
    for i in range(n):
        out.append(Unit(
            key=f"{prefix}{i + 1}", name=f"{name}{i + 1}", group=group, fuel=fuel,
            pmax=pmax, pmin=round(pmax * pmin_ratio, 1), eff=round(eff0 - eff_step * i, 4), om=om,
            ramp=round(pmax * ramp_ratio, 1), min_up=ut, min_down=dt, startup=round(su_per * pmax), fast=fast,
        ))
    return out


def default_units() -> list[Unit]:
    """東京エリアの火力構成を大まかに模した仮想電源（合計 約 4,300 万kW）"""
    u: list[Unit] = []
    u += _units("coalU", 2, "石炭(超々臨界)", "coal", "coal", 100, 0.40, 0.430, 0.003, 0.8, 0.40, 8, 8, 15)
    u += _units("coalS", 2, "石炭(超臨界)", "coal", "coal", 100, 0.40, 0.410, 0.003, 0.8, 0.40, 8, 8, 15)
    u += _units("coalO", 2, "石炭(亜臨界)", "coal", "coal", 70, 0.40, 0.380, 0.003, 0.9, 0.40, 8, 8, 15)
    u += _units("cc16", 4, "LNG複合(1600℃級)", "lng", "lng", 200, 0.30, 0.620, 0.004, 0.3, 0.80, 3, 3, 1.5)
    u += _units("cc13", 4, "LNG複合(1300℃級)", "lng", "lng", 250, 0.30, 0.560, 0.004, 0.3, 0.80, 3, 3, 1.5)
    u += _units("cc11", 3, "LNG複合(1100℃級)", "lng", "lng", 250, 0.30, 0.500, 0.004, 0.4, 0.80, 3, 3, 1.5)
    u += _units("lngS", 3, "LNG汽力", "lng", "lng", 230, 0.25, 0.400, 0.005, 0.5, 0.60, 6, 6, 3.0)
    u += _units("oil", 2, "石油火力", "oil", "oil", 175, 0.20, 0.380, 0.005, 0.8, 0.80, 4, 4, 2.0)
    u += _units("gt", 1, "緊急設置GT", "oil", "lng", 150, 0.10, 0.330, 0.0, 1.0, 1.00, 1, 1, 0.3, fast=True)
    for x in u:
        if x.key == "gt1":
            x.name = "緊急設置ガスタービン"
            x.group = "oil"
    return u


GROUPS = [  # (key, 表示名) ― 積み上げ順（下から）
    ("base", "原子力・水力"),
    ("coal", "石炭"),
    ("lng", "LNG火力"),
    ("oil", "石油・GT"),
    ("import", "連系線受電"),
    ("pumped", "揚水発電"),
    ("solar", "太陽光"),
]


@dataclass
class Scenario:
    lng_mult: float = 1.0
    coal_mult: float = 1.0
    oil_mult: float = 1.0
    co2_price: float = 0.0       # 円/t-CO2
    solar_mult: float = 1.0      # 太陽光導入量の倍率
    reserve_pct: float = 3.0     # 運転予備率（%）
    nuclear_on: bool = False     # 原子力 1 基（135 万kW）を稼働させるか
    nuclear_cap: float = 135.0
    nuclear_cost: float = 1.8    # 円/kWh
    hydro: float = 150.0         # 一般水力（万kW、一定）
    import_cap: float = 400.0    # 連系線受電の上限（万kW）
    import_price: float = 14.0   # 円/kWh
    ps_gen_cap: float = 900.0    # 揚水 発電上限（万kW）
    ps_pump_cap: float = 900.0   # 揚水 揚水動力上限（万kW）
    ps_energy: float = 6000.0    # 揚水 上池容量（万kWh）
    ps_eff: float = 0.70         # 揚水 総合効率
    voll: float = 300.0          # 供給不足ペナルティ（円/kWh）
    curtail_cost: float = 0.5    # 太陽光出力制御のペナルティ（円/kWh）
    mip_gap: float = 0.002
    time_limit: float = 60.0


@dataclass
class DispatchResult:
    status: str
    objective: float                      # 万円
    solve_seconds: float
    demand: np.ndarray
    solar_avail: np.ndarray
    unit_keys: list
    output: dict = field(default_factory=dict)   # key -> (T,)
    on: dict = field(default_factory=dict)       # key -> (T,) 0/1
    starts: dict = field(default_factory=dict)   # key -> 回数
    groups: dict = field(default_factory=dict)   # group -> (T,)
    ps_gen: np.ndarray | None = None
    ps_pump: np.ndarray | None = None
    soc: np.ndarray | None = None
    imports: np.ndarray | None = None
    solar_used: np.ndarray | None = None
    curtail: np.ndarray | None = None
    unserved: np.ndarray | None = None
    surplus: np.ndarray | None = None
    nuclear: np.ndarray | None = None
    hydro: np.ndarray | None = None
    price: np.ndarray | None = None              # 円/kWh
    cost_by_group: dict = field(default_factory=dict)   # 万円
    startup_cost: float = 0.0
    co2_t: float = 0.0
    gap: float | None = None


def _solver(sc: Scenario, msg=False, mip=True):
    kw = dict(msg=msg, timeLimit=sc.time_limit)
    if mip:
        kw["gapRel"] = sc.mip_gap
    return pulp.HiGHS(**kw)


def _build(demand, solar, units, sc, fixed_on=None, fixed_mode=None, relax=False):
    T = len(demand)
    H = range(T)
    prev = lambda t: (t - 1) % T  # 1 日周期（24 時の状態が翌 0 時へつながる）
    prob = pulp.LpProblem("tokyo_dispatch", pulp.LpMinimize)
    act = [g for g in units if g.enabled]

    p, u, v, w = {}, {}, {}, {}
    for g in act:
        for t in H:
            p[g.key, t] = pulp.LpVariable(f"p_{g.key}_{t}", 0, g.pmax)
            fixed = None
            if fixed_on is not None and g.key in fixed_on:
                fixed = float(round(fixed_on[g.key][t]))
            if fixed is not None:
                u[g.key, t] = pulp.LpVariable(f"u_{g.key}_{t}", fixed, fixed)
            elif relax:
                u[g.key, t] = pulp.LpVariable(f"u_{g.key}_{t}", 0, 1)
            else:
                u[g.key, t] = pulp.LpVariable(f"u_{g.key}_{t}", cat="Binary")
            v[g.key, t] = pulp.LpVariable(f"v_{g.key}_{t}", 0, 1)
            w[g.key, t] = pulp.LpVariable(f"w_{g.key}_{t}", 0, 1)

    S = np.asarray(solar, float) * sc.solar_mult
    gen = {t: pulp.LpVariable(f"psg_{t}", 0, sc.ps_gen_cap) for t in H}
    pump = {t: pulp.LpVariable(f"psp_{t}", 0, sc.ps_pump_cap) for t in H}
    soc = {t: pulp.LpVariable(f"soc_{t}", 0, sc.ps_energy) for t in H}
    mode = {}
    for t in H:
        if fixed_mode is not None:
            m = float(round(fixed_mode[t]))
            mode[t] = pulp.LpVariable(f"mode_{t}", m, m)
        elif relax:
            mode[t] = pulp.LpVariable(f"mode_{t}", 0, 1)
        else:
            mode[t] = pulp.LpVariable(f"mode_{t}", cat="Binary")
    imp = {t: pulp.LpVariable(f"imp_{t}", 0, sc.import_cap) for t in H}
    sv = {t: pulp.LpVariable(f"solar_{t}", 0, max(float(S[t]), 0.0)) for t in H}
    ens = {t: pulp.LpVariable(f"ens_{t}", 0) for t in H}
    dump = {t: pulp.LpVariable(f"dump_{t}", 0) for t in H}
    nuc = sc.nuclear_cap if sc.nuclear_on else 0.0

    # ---- 目的関数（万円）
    cost = {g.key: g.var_cost(sc) for g in act}
    obj = pulp.lpSum(cost[g.key] * p[g.key, t] + g.startup * v[g.key, t] for g in act for t in H)
    obj += pulp.lpSum(sc.import_price * imp[t] for t in H)
    obj += pulp.lpSum(sc.curtail_cost * (S[t] - sv[t]) for t in H)
    obj += pulp.lpSum(sc.voll * (ens[t] + dump[t]) for t in H)
    obj += sc.nuclear_cost * nuc * T
    prob += obj

    # ---- 需給バランス（双対値＝限界価格）
    bal = {}
    for t in H:
        c = (pulp.lpSum(p[g.key, t] for g in act) + nuc + sc.hydro + gen[t] - pump[t]
             + imp[t] + sv[t] + ens[t] - dump[t] == float(demand[t]))
        prob += c, f"balance_{t}"
        bal[t] = c

    # ---- 運転予備力：運転中火力の余力＋揚水の発電余力 ≥ 需要×予備率
    for t in H:
        prob += (pulp.lpSum(g.pmax * u[g.key, t] - p[g.key, t] for g in act)
                 + (sc.ps_gen_cap - gen[t]) >= sc.reserve_pct / 100.0 * float(demand[t])), f"reserve_{t}"

    # ---- 火力ユニット制約
    for g in act:
        for t in H:
            k = g.key
            prob += p[k, t] <= g.pmax * u[k, t]
            prob += p[k, t] >= g.pmin * u[k, t]
            prob += u[k, t] - u[k, prev(t)] == v[k, t] - w[k, t]
            prob += p[k, t] - p[k, prev(t)] <= g.ramp * u[k, prev(t)] + g.pmin * v[k, t]
            prob += p[k, prev(t)] - p[k, t] <= g.ramp * u[k, t] + g.pmin * w[k, t]
            if g.min_up > 1:
                prob += pulp.lpSum(v[k, (t - j) % T] for j in range(g.min_up)) <= u[k, t]
            if g.min_down > 1:
                prob += pulp.lpSum(w[k, (t - j) % T] for j in range(g.min_down)) <= 1 - u[k, t]

    # ---- 揚水発電（1 日周期で貯水量が元に戻る）
    eta = sc.ps_eff ** 0.5
    for t in H:
        prob += gen[t] <= sc.ps_gen_cap * mode[t]
        prob += pump[t] <= sc.ps_pump_cap * (1 - mode[t])
        prob += soc[t] == soc[prev(t)] + eta * pump[t] - gen[t] * (1.0 / eta)

    vars_ = dict(p=p, u=u, v=v, gen=gen, pump=pump, soc=soc, mode=mode, imp=imp, sv=sv, ens=ens, dump=dump,
                 bal=bal, nuc=nuc, S=S, act=act, cost=cost)
    return prob, vars_


def solve_dispatch(demand, solar, units=None, sc: Scenario | None = None, fixed_on=None,
                   with_prices=True, msg=False) -> DispatchResult:
    """
    fixed_on: {unit_key: (T,) 0/1} を与えると、その電源の起動停止を固定して解く（当日再給電の評価用）
    with_prices: MILP の解で起動停止を固定した LP を解き、需給バランスの双対値から限界価格を求める
    """
    units = units if units is not None else default_units()
    sc = sc or Scenario()
    demand = np.asarray(demand, float)
    T = len(demand)
    t0 = time.perf_counter()
    prob, V = _build(demand, solar, units, sc, fixed_on=fixed_on)
    prob.solve(_solver(sc, msg=msg, mip=True))
    status = pulp.LpStatus[prob.status]
    act = V["act"]
    val = lambda x: float(x.varValue or 0.0)

    res = DispatchResult(status=status, objective=float(pulp.value(prob.objective) or np.nan),
                         solve_seconds=0.0, demand=demand, solar_avail=V["S"], unit_keys=[g.key for g in act])
    if status not in ("Optimal",) and prob.status != 1:
        res.solve_seconds = time.perf_counter() - t0
        return res

    for g in act:
        res.output[g.key] = np.array([val(V["p"][g.key, t]) for t in range(T)])
        res.on[g.key] = np.array([round(val(V["u"][g.key, t])) for t in range(T)], dtype=float)
        res.starts[g.key] = int(round(sum(val(V["v"][g.key, t]) for t in range(T))))
    res.ps_gen = np.array([val(V["gen"][t]) for t in range(T)])
    res.ps_pump = np.array([val(V["pump"][t]) for t in range(T)])
    res.soc = np.array([val(V["soc"][t]) for t in range(T)])
    res.imports = np.array([val(V["imp"][t]) for t in range(T)])
    res.solar_used = np.array([val(V["sv"][t]) for t in range(T)])
    res.curtail = np.maximum(V["S"] - res.solar_used, 0.0)
    res.unserved = np.array([val(V["ens"][t]) for t in range(T)])
    res.surplus = np.array([val(V["dump"][t]) for t in range(T)])
    res.nuclear = np.full(T, V["nuc"])
    res.hydro = np.full(T, sc.hydro)
    mode = np.array([round(val(V["mode"][t])) for t in range(T)], dtype=float)

    # 区分別の出力
    grp = {k: np.zeros(T) for k, _ in GROUPS}
    grp["base"] += res.nuclear + res.hydro
    for g in act:
        grp[g.group] += res.output[g.key]
    grp["import"] += res.imports
    grp["pumped"] += res.ps_gen
    grp["solar"] += res.solar_used
    res.groups = grp

    # 費用・CO2
    cbg = {k: 0.0 for k, _ in GROUPS}
    co2 = 0.0
    for g in act:
        e = res.output[g.key].sum()                         # 万kWh
        cbg[g.group] += V["cost"][g.key] * e                # 万円
        co2 += g.co2() * e * 1e4 / 1000.0                   # t-CO2
    cbg["import"] += sc.import_price * res.imports.sum()
    cbg["base"] += sc.nuclear_cost * res.nuclear.sum()
    res.cost_by_group = cbg
    res.startup_cost = float(sum(g.startup * res.starts[g.key] for g in act))
    res.co2_t = co2
    res.solve_seconds = time.perf_counter() - t0

    if with_prices:
        try:
            lp, LV = _build(demand, solar, units, sc, fixed_on=res.on, fixed_mode=mode, relax=True)
            lp.solve(_solver(sc, msg=False, mip=False))
            if lp.status == 1:
                res.price = np.array([float(LV["bal"][t].pi or 0.0) for t in range(T)])
        except Exception:  # 価格計算に失敗しても計画結果は返す
            res.price = None
        res.solve_seconds = time.perf_counter() - t0
    return res


@dataclass
class ForecastValue:
    plan: DispatchResult        # 予測需要で立てた前日計画
    realtime: DispatchResult    # 実績需要に対し、前日計画の起動停止を固定して再給電
    perfect: DispatchResult     # 実績需要を完全に知っていた場合の最適計画

    @property
    def extra_cost(self) -> float:
        return self.realtime.objective - self.perfect.objective


def evaluate_forecast_value(demand_fc, demand_act, solar, units=None, sc=None) -> ForecastValue:
    """予測誤差が発電コストに与える影響（＝予測精度の経済価値）を評価する。"""
    units = units if units is not None else default_units()
    sc = sc or Scenario()
    plan = solve_dispatch(demand_fc, solar, units, sc, with_prices=True)
    fixed = {k: v for k, v in plan.on.items() if not next(g for g in units if g.key == k).fast}
    rt = solve_dispatch(demand_act, solar, units, sc, fixed_on=fixed, with_prices=False)
    pf = solve_dispatch(demand_act, solar, units, sc, with_prices=False)
    return ForecastValue(plan, rt, pf)


def clone_units(units):
    return copy.deepcopy(units)
