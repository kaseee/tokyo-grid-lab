"""アプリ同梱データの読み込みと集計ヘルパ"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def resource_dir() -> Path:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def asset(*parts: str) -> Path:
    return resource_dir().joinpath("assets", *parts)


def fmt_int(x: float) -> str:
    return f"{x:,.0f}"


class DataStore:
    def __init__(self) -> None:
        d = np.load(asset("app_data.npz"), allow_pickle=False)
        self.ts = d["ts"].astype("datetime64[h]")
        self.day = self.ts.astype("datetime64[D]")
        self.month = self.ts.astype("datetime64[M]")
        self.hour = (self.ts - self.day).astype(int)
        self.demand = d["demand"].astype(float)
        self.pred_initial = d["pred"].astype(float)
        self.pred = self.pred_initial.copy()
        self.tepco = d["tepco_forecast"].astype(float)
        self.solar = d["solar"].astype(float)
        self.temp = d["temp"].astype(float)
        self.rh = d["rh"].astype(float)
        self.sun = d["sun"].astype(float)
        self.X = d["X"]
        self.feature_names = [str(x) for x in d["feature_names"]]
        self.feature_labels = [str(x) for x in d["feature_labels"]]
        self.feature_groups = [str(x) for x in d["feature_groups"]]
        self.m_fit = d["mask_fit"].astype(bool)
        self.m_valid = d["mask_valid"].astype(bool)
        self.m_train = d["mask_train"].astype(bool)
        self.m_test = d["mask_test"].astype(bool)
        with open(asset("metrics.json"), encoding="utf-8") as f:
            self.metrics = json.load(f)
        self.test_days = np.unique(self.day[self.m_test])
        self.daytype = self.X[:, self.feature_names.index("day_type")].astype(int)
        self.naive = self.X[:, self.feature_names.index("lag168")].astype(float)

    # ---- 指定日の 24 時間データ
    def day_mask(self, day) -> np.ndarray:
        return self.day == np.datetime64(day, "D")

    def day_series(self, day):
        m = self.day_mask(day)
        return self.demand[m], self.pred[m], self.solar[m], self.temp[m]

    # ---- テスト期間の集計
    def test_metrics(self, pred=None) -> dict:
        from core.forecast import metrics
        p = self.pred if pred is None else pred
        t = self.m_test
        return metrics(self.demand[t], p[t], self.day[t])

    def monthly_mape(self, pred=None):
        p = self.pred if pred is None else pred
        t = self.m_test
        months = self.month[t]
        ape = np.abs(p[t] - self.demand[t]) / self.demand[t] * 100
        um = np.unique(months)
        return um, np.array([ape[months == m].mean() for m in um])

    def hourly_mape(self, pred=None):
        p = self.pred if pred is None else pred
        t = self.m_test
        ape = np.abs(p[t] - self.demand[t]) / self.demand[t] * 100
        h = self.hour[t]
        return np.array([ape[h == k].mean() for k in range(24)])

    def daily_profile(self):
        """全期間の日平均・日最小・日最大需要（概要グラフ用）"""
        days, inv = np.unique(self.day, return_inverse=True)
        n = len(days)
        mean = np.bincount(inv, weights=self.demand, minlength=n) / np.bincount(inv, minlength=n)
        mx = np.full(n, -np.inf)
        mn = np.full(n, np.inf)
        np.maximum.at(mx, inv, self.demand)
        np.minimum.at(mn, inv, self.demand)
        return days, mean, mn, mx

    def notable_days(self) -> dict:
        """テスト期間の代表日（夏・冬の最大需要日、春の軽負荷日）"""
        t = self.m_test
        days = self.day[t]
        dem = self.demand[t]
        net = dem - self.solar[t]
        months = self.month[t].astype(object)
        out = {}
        summer = np.isin(np.array([m.month for m in months]), [7, 8, 9])
        winter = np.isin(np.array([m.month for m in months]), [12, 1, 2])
        spring = np.isin(np.array([m.month for m in months]), [3, 4, 5])
        if summer.any():
            out["夏の最大需要日"] = days[summer][np.argmax(dem[summer])]
        if winter.any():
            out["冬の最大需要日"] = days[winter][np.argmax(dem[winter])]
        if spring.any():
            out["春の軽負荷日"] = days[spring][np.argmin(net[spring])]
        return out
