"""
需要予測コア（numpy + LightGBM のみ依存。exe からも同じコードで再学習する）
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

try:  # exe 起動を軽くするため遅延 import も可能にしておく
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None

DEFAULT_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}


@dataclass
class TrainResult:
    booster: object
    feature_names: list
    best_iter: int
    valid_mape: float
    train_seconds: float
    params: dict = field(default_factory=dict)


def mape(y, p) -> float:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p) & (y > 0)
    return float(np.mean(np.abs(p[m] - y[m]) / y[m]) * 100.0)


def metrics(y, p, day_index=None) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    m = np.isfinite(y) & np.isfinite(p) & (y > 0)
    y, p = y[m], p[m]
    err = p - y
    out = {
        "MAPE": float(np.mean(np.abs(err) / y) * 100.0),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1.0 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
        "bias": float(np.mean(err)),
        "n": int(len(y)),
    }
    if day_index is not None:
        d = np.asarray(day_index)[m]
        uniq, inv = np.unique(d, return_inverse=True)
        ymax = np.full(len(uniq), -np.inf)
        pmax = np.full(len(uniq), -np.inf)
        np.maximum.at(ymax, inv, y)
        np.maximum.at(pmax, inv, p)
        out["peak_MAPE"] = float(np.mean(np.abs(pmax - ymax) / ymax) * 100.0)
    return out


def train_lgbm(X, y, feature_names, cols, fit_mask, valid_mask, refit_mask=None,
               params=None, num_boost_round=4000, early_stopping=150, log_target=True,
               progress=None) -> TrainResult:
    """
    fit_mask で学習し valid_mask で early stopping → 最適ラウンド数を決定。
    refit_mask が与えられたら、そのラウンド数(×1.1)で refit_mask 全体を再学習する。
    log_target=True のとき log(需要) を学習（相対誤差＝MAPE に近い損失になる）。
    """
    if lgb is None:
        raise RuntimeError("lightgbm が読み込めません")
    t0 = time.perf_counter()
    prm = dict(DEFAULT_PARAMS)
    if params:
        prm.update(params)
    cols = list(cols)
    names = [feature_names[i] for i in cols]
    Xc = X[:, cols]
    tf = (lambda v: np.log(v)) if log_target else (lambda v: v)

    dfit = lgb.Dataset(Xc[fit_mask], tf(y[fit_mask]), feature_name=names, free_raw_data=False)
    cbs = []
    if progress is not None:
        def _cb(env):
            if env.iteration % 25 == 0:
                progress(env.iteration, env.end_iteration)
        cbs.append(_cb)
    if valid_mask is None:  # 本数を固定して学習（early stopping なし）
        b = lgb.train(prm, dfit, num_boost_round=num_boost_round, callbacks=cbs)
        best, vmape = num_boost_round, float("nan")
    else:
        dval = lgb.Dataset(Xc[valid_mask], tf(y[valid_mask]), reference=dfit, feature_name=names)
        cbs.append(lgb.early_stopping(early_stopping, verbose=False))
        b = lgb.train(prm, dfit, num_boost_round=num_boost_round, valid_sets=[dval], callbacks=cbs)
        best = int(b.best_iteration or num_boost_round)
        pv = b.predict(Xc[valid_mask], num_iteration=best)
        if log_target:
            pv = np.exp(pv)
        vmape = mape(y[valid_mask], pv)

    if refit_mask is not None:
        n_final = int(best * 1.1)
        dall = lgb.Dataset(Xc[refit_mask], tf(y[refit_mask]), feature_name=names)
        b = lgb.train(prm, dall, num_boost_round=n_final)
        best = n_final
    return TrainResult(b, names, best, vmape, time.perf_counter() - t0, prm)


def predict(result_or_booster, X, cols=None, log_target=True):
    b = result_or_booster.booster if isinstance(result_or_booster, TrainResult) else result_or_booster
    Xc = X if cols is None else X[:, list(cols)]
    p = b.predict(Xc, num_iteration=getattr(result_or_booster, "best_iter", None))
    return np.exp(p) if log_target else p


def importance(booster, kind="gain"):
    names = booster.feature_name()
    vals = booster.feature_importance(importance_type=kind).astype(float)
    tot = vals.sum() or 1.0
    order = np.argsort(-vals)
    return [(names[i], float(vals[i] / tot * 100.0)) for i in order]
