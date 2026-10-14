"""
需要予測用の特徴量（前日予測＝day-ahead を想定）

想定する予測タイミング：対象日 D の前日 D-1 の午前。
  - カレンダー：対象日の曜日・祝日など（事前に確定）
  - 気象      ：対象日の気温・湿度・日照（※実測値を「完全な気象予報」の代わりに使用）
  - ラグ      ：D-1 午前時点で確定している需要（2日前・7日前の同時刻など）
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import jpholiday
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from core.features_meta import (ALL_FEATURES, FEATURE_GROUPS, FEATURE_LABELS_JA,  # noqa: E402,F401
                                GROUP_LABELS_JA)


def _day_type(d: dt.date) -> int:
    """0: 平日, 1: 土曜, 2: 日曜・祝日・年末年始"""
    if jpholiday.is_holiday(d) or (d.month == 12 and d.day >= 29) or (d.month == 1 and d.day <= 3):
        return 2
    if d.weekday() == 6:
        return 2
    if d.weekday() == 5:
        return 1
    return 0


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df: index=datetime(時間), 列 demand / temp_* / rh_* / sun_* を含む"""
    X = pd.DataFrame(index=df.index)
    idx = df.index
    dates = idx.normalize()
    uniq = pd.DatetimeIndex(dates.unique())
    ext = pd.date_range(uniq.min() - pd.Timedelta(days=8), uniq.max() + pd.Timedelta(days=1), freq="D")
    dtype_map = pd.Series([_day_type(d.date()) for d in ext], index=ext)
    hol_map = pd.Series([int(jpholiday.is_holiday(d.date())) for d in ext], index=ext)

    # ---- カレンダー
    X["hour"] = idx.hour
    X["dow"] = idx.dayofweek
    X["month"] = idx.month
    doy = idx.dayofyear.values
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X["is_holiday"] = hol_map.reindex(dates).values
    X["day_type"] = dtype_map.reindex(dates).values
    md = idx.month * 100 + idx.day
    X["is_obon"] = ((md >= 813) & (md <= 816)).astype(int)
    X["is_newyear"] = ((md >= 1229) | (md <= 103)).astype(int)
    X["is_gw"] = ((md >= 429) & (md <= 506)).astype(int)
    prev_t = dtype_map.reindex(dates - pd.Timedelta(days=1)).values
    next_t = dtype_map.reindex(dates + pd.Timedelta(days=1)).values
    X["day_type_prev"] = prev_t
    X["day_type_next"] = next_t
    X["is_bridge"] = ((X["day_type"] == 0) & (prev_t == 2) & (next_t == 2)).astype(int)

    # ---- 気象（対象日の値＝予報値の代わりに実測値）
    X["temp_kanto"] = df["temp_kanto"]
    X["temp_tokyo"] = df["temp_tokyo"]
    X["rh_kanto"] = df["rh_kanto"]
    X["sun_kanto"] = df["sun_kanto"]
    t, h = df["temp_kanto"], df["rh_kanto"]
    X["thi"] = 0.81 * t + 0.01 * h * (0.99 * t - 14.3) + 46.3
    g = df["temp_kanto"].groupby(dates)
    X["temp_day_max"] = g.transform("max")
    X["temp_day_min"] = g.transform("min")
    X["temp_day_mean"] = g.transform("mean")
    day_mean = g.mean()
    X["temp_prev_day_mean"] = day_mean.reindex(dates - pd.Timedelta(days=1)).values
    X["temp_roll3"] = df["temp_kanto"].rolling(3, min_periods=1, center=True).mean()
    X["sun_day_total"] = df["sun_kanto"].groupby(dates).transform("sum")
    X["cdd"] = np.maximum(df["temp_kanto"] - 22.0, 0.0)
    X["hdd"] = np.maximum(14.0 - df["temp_kanto"], 0.0)

    # ---- 需要ラグ（D-1 午前時点で確定している情報のみ）
    dem = df["demand"]
    X["lag48"] = dem.shift(48)
    X["lag168"] = dem.shift(168)
    dg = dem.groupby(dates)
    dmean, dmax = dg.mean(), dg.max()
    X["day_mean_lag2"] = dmean.reindex(dates - pd.Timedelta(days=2)).values
    X["day_max_lag2"] = dmax.reindex(dates - pd.Timedelta(days=2)).values
    X["day_type_lag2"] = dtype_map.reindex(dates - pd.Timedelta(days=2)).values
    X["day_type_lag7"] = dtype_map.reindex(dates - pd.Timedelta(days=7)).values

    return X[ALL_FEATURES].astype("float32")
