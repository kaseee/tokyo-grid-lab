"""
LightGBM による東京エリア電力需要（1時間値）の前日予測モデルを学習・評価し、
可視化アプリ用のデータ（app/assets/）を書き出す。

期間設定
  学習      : 2022-04-08 〜 2025-09-30（うち 2025-04-01 以降を early stopping 用の検証に使用）
  テスト    : 2025-10-01 〜 2026-10-12（直近約 1 年、学習に一切使わない）
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "app"))

from features import (ALL_FEATURES, FEATURE_GROUPS, FEATURE_LABELS_JA,  # noqa: E402
                      GROUP_LABELS_JA, build_features)
from core.forecast import importance, metrics, predict, train_lgbm  # noqa: E402

ASSETS = ROOT / "app" / "assets"
FIT_END = "2025-04-01"
TRAIN_END = "2025-10-01"
MODEL_PARAMS = {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 40}
LOG_TARGET = True


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "data/processed/hourly_dataset.csv", parse_dates=["datetime"], index_col="datetime")
    X = build_features(df)
    y = df["demand"].to_numpy(float)
    idx = df.index
    ok = X["lag168"].notna().to_numpy() & np.isfinite(y)
    train = ok & (idx < TRAIN_END)
    fit = train & (idx < FIT_END)
    val = train & (idx >= FIT_END)
    test = ok & (idx >= TRAIN_END)
    Xa = X.to_numpy(np.float32)
    names = list(X.columns)
    day = idx.normalize().to_numpy()
    allc = list(range(len(names)))

    print(f"学習 {train.sum():,} 行 / 検証(early stopping) {val.sum():,} 行 / テスト {test.sum():,} 行")

    # ---- 本番モデル
    res = train_lgbm(Xa, y, names, allc, fit, val, refit_mask=train, params=MODEL_PARAMS, log_target=LOG_TARGET)
    pred = predict(res, Xa, allc, log_target=LOG_TARGET).astype(np.float32)
    m_test = metrics(y[test], pred[test], day[test])
    m_train = metrics(y[train], pred[train], day[train])
    print(f"テスト MAPE {m_test['MAPE']:.3f}%  ピーク MAPE {m_test['peak_MAPE']:.3f}%  "
          f"MAE {m_test['MAE']:.1f}万kW  R2 {m_test['R2']:.4f}  ({res.best_iter} rounds, {res.train_seconds:.1f}s)")
    (ROOT / "models").mkdir(exist_ok=True)
    res.booster.save_model(str(ROOT / "models" / "lgbm_model.txt"), num_iteration=res.best_iter)

    # ---- ベースライン
    naive = X["lag168"].to_numpy(float)
    m_naive = metrics(y[test], naive[test], day[test])
    tepco = df["tepco_forecast"].to_numpy(float)
    m_tepco = metrics(y[test], tepco[test], day[test])
    print(f"ベースライン：前週同時刻 MAPE {m_naive['MAPE']:.3f}% / 東電 当日予測値(参考) {m_tepco['MAPE']:.3f}%")

    # ---- 特徴量グループのアブレーション
    gidx = {k: [names.index(f) for f in v] for k, v in FEATURE_GROUPS.items()}
    ablation = []
    for key, cols in [("calendar", gidx["calendar"]),
                      ("calendar+lag", gidx["calendar"] + gidx["lag"]),
                      ("calendar+weather", gidx["calendar"] + gidx["weather"]),
                      ("all", allc)]:
        if key == "all":
            mm = m_test
        else:
            r = train_lgbm(Xa, y, names, cols, fit, val, refit_mask=train, params=MODEL_PARAMS, log_target=LOG_TARGET)
            p = predict(r, Xa, cols, log_target=LOG_TARGET)
            mm = metrics(y[test], p[test], day[test])
        label = " + ".join(GROUP_LABELS_JA[k] for k in key.split("+")) if key != "all" else "全特徴量"
        ablation.append({"key": key, "label": label, "MAPE": mm["MAPE"], "peak_MAPE": mm["peak_MAPE"]})
        print(f"  アブレーション {label:16s} MAPE {mm['MAPE']:.3f}%")

    # ---- 集計
    t_idx = idx[test]
    ape = np.abs(pred[test] - y[test]) / y[test] * 100
    s = pd.Series(ape, index=t_idx)
    monthly = s.groupby(t_idx.strftime("%Y-%m")).mean()
    hourly = s.groupby(t_idx.hour).mean()
    dtype = pd.Series(X["day_type"].to_numpy()[test], index=t_idx)
    by_dtype = s.groupby(dtype.values).mean()
    imp = importance(res.booster, "gain")

    out = {
        "periods": {
            "data_start": str(idx.min()), "data_end": str(idx.max()),
            "train_start": str(idx[train].min()), "train_end": str(idx[train].max()),
            "fit_end": FIT_END, "test_start": str(idx[test].min()), "test_end": str(idx[test].max()),
            "n_train": int(train.sum()), "n_valid": int(val.sum()), "n_test": int(test.sum()),
        },
        "model": {"params": {**MODEL_PARAMS, "log_target": LOG_TARGET}, "best_iter": res.best_iter,
                  "valid_MAPE": res.valid_mape, "train_seconds": res.train_seconds},
        "test": m_test, "train": m_train,
        "baselines": {"naive_week": m_naive, "tepco_intraday": m_tepco},
        "ablation": ablation,
        "monthly_MAPE": {k: float(v) for k, v in monthly.items()},
        "hourly_MAPE": [float(hourly.get(h, np.nan)) for h in range(24)],
        "daytype_MAPE": {str(int(k)): float(v) for k, v in by_dtype.items()},
        "importance_gain": [{"name": n, "label": FEATURE_LABELS_JA.get(n, n), "value": v} for n, v in imp],
    }
    with open(ASSETS / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # ---- アプリ用データ
    group_of = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
    np.savez_compressed(
        ASSETS / "app_data.npz",
        ts=idx.to_numpy().astype("datetime64[h]"),
        demand=y.astype(np.float32),
        pred=pred,
        tepco_forecast=tepco.astype(np.float32),
        solar=df["solar"].to_numpy(np.float32),
        temp=df["temp_kanto"].to_numpy(np.float32),
        rh=df["rh_kanto"].to_numpy(np.float32),
        sun=df["sun_kanto"].to_numpy(np.float32),
        X=Xa,
        feature_names=np.array(names),
        feature_labels=np.array([FEATURE_LABELS_JA.get(n, n) for n in names]),
        feature_groups=np.array([group_of[n] for n in names]),
        mask_fit=fit, mask_valid=val, mask_train=train, mask_test=test,
    )
    pd.DataFrame({"actual": y[test], "lightgbm": pred[test], "naive_week": naive[test],
                  "tepco_intraday": tepco[test]}, index=t_idx).to_csv(ROOT / "data/processed/test_predictions.csv",
                                                                      float_format="%.1f")
    print("saved:", ASSETS)


if __name__ == "__main__":
    main()
