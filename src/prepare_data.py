"""
公開データの前処理
  - 東京電力パワーグリッド「でんき予報」過去の電力使用実績（月別zip → 日別CSV）
  - 気象庁「過去の気象データ・ダウンロード」時別値（関東8地点：気温・相対湿度・日照時間）
を 1 時間単位のデータセットに整形して data/processed/hourly_dataset.csv に保存する。

時刻の扱い：
  東電の「0:00」行 = 0:00〜1:00 の 1 時間平均需要。
  気象庁の時別値（気温・湿度）は正時の瞬間値、日照時間は「前 1 時間」の積算値。
  → 需要 [h, h+1) に対し、気温・湿度は h と h+1 の平均、日照は h+1 時の値を対応させる。
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"
OUT = ROOT / "data" / "processed"

# 気象官署コード → (列名サフィックス, 地点名, 都県人口[2020年国勢調査, 万人])
STATIONS = {
    "東京": ("tokyo", 1404.8),
    "横浜": ("yokohama", 923.7),
    "熊谷": ("kumagaya", 734.5),
    "千葉": ("chiba", 628.4),
    "水戸": ("mito", 286.7),
    "宇都宮": ("utsunomiya", 193.3),
    "前橋": ("maebashi", 193.9),
    "甲府": ("kofu", 81.0),
}


# ----------------------------------------------------------------------------
# 東京電力 でんき予報
# ----------------------------------------------------------------------------
def _parse_tepco_day(text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    lines = text.splitlines()
    hourly, five = [], []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("DATE,TIME,当日実績(万kW)"):
            for row in lines[i + 1 : i + 25]:
                d, t, act, fc, rate, sup = row.split(",")[:6]
                hourly.append((d, t, act, fc, rate, sup))
            i += 25
            continue
        if line.startswith("DATE,TIME,当日実績(５分間隔値)"):
            for row in lines[i + 1 : i + 289]:
                if not row.strip():
                    break
                d, t, act, pv, pct = row.split(",")[:5]
                five.append((d, t, act, pv))
            i += 289
            continue
        i += 1
    h = pd.DataFrame(hourly, columns=["date", "time", "demand", "tepco_forecast", "usage_rate", "supply"])
    f = pd.DataFrame(five, columns=["date", "time", "demand_5min", "solar_5min"])
    return h, f


def _latest(pattern: str) -> Path:
    files = sorted(RAW.glob(pattern))
    if not files:
        raise FileNotFoundError(f"data_raw/{pattern} がありません。先に python tools/download_data.py を実行してください。")
    return files[-1]


def load_tepco() -> pd.DataFrame:
    outer = zipfile.ZipFile(_latest("tepco_power_usage_*.zip"))
    hs, fs = [], []
    for name in sorted(outer.namelist()):
        inner = zipfile.ZipFile(io.BytesIO(outer.read(name)))
        for n in sorted(inner.namelist()):
            h, f = _parse_tepco_day(inner.read(n).decode("cp932"))
            hs.append(h)
            fs.append(f)
    h = pd.concat(hs, ignore_index=True)
    f = pd.concat(fs, ignore_index=True)

    h["datetime"] = pd.to_datetime(h["date"] + " " + h["time"], format="%Y/%m/%d %H:%M")
    for c in ["demand", "tepco_forecast", "usage_rate", "supply"]:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    # 未確定値・欠測は 0 で入っているので NaN にする
    h.loc[h["demand"] <= 0, "demand"] = np.nan
    h.loc[h["tepco_forecast"] <= 0, "tepco_forecast"] = np.nan
    h = h.drop(columns=["date", "time"]).set_index("datetime").sort_index()

    f["datetime"] = pd.to_datetime(f["date"] + " " + f["time"], format="%Y/%m/%d %H:%M")
    for c in ["demand_5min", "solar_5min"]:
        f[c] = pd.to_numeric(f[c], errors="coerce")
    f = f.set_index("datetime").sort_index()
    f5 = f[["demand_5min", "solar_5min"]].resample("1h").mean()
    f5.columns = ["demand_from_5min", "solar"]

    df = h.join(f5, how="left")
    return df


# ----------------------------------------------------------------------------
# 気象庁 時別値
# ----------------------------------------------------------------------------
def _parse_jma_csv(text: str) -> pd.DataFrame:
    rows = text.splitlines()
    # 0: ダウンロードした時刻, 1: 空行, 2: 地点名, 3: 要素名, 4: 付帯情報, 5〜: データ
    stations = rows[2].split(",")
    elements = rows[3].split(",")
    subs = rows[4].split(",")
    cols = {}
    for j in range(1, len(stations)):
        st, el, sub = stations[j], elements[j], subs[j]
        if sub != "":  # 品質情報・均質番号・現象なし情報は使わない
            continue
        if el.startswith("気温"):
            key = "temp"
        elif el.startswith("相対湿度"):
            key = "rh"
        elif el.startswith("日照時間"):
            key = "sun"
        else:
            continue
        cols[j] = f"{key}_{STATIONS[st][0]}"
    data = [r.split(",") for r in rows[5:] if r.strip()]
    out = pd.DataFrame({name: [d[j] for d in data] for j, name in cols.items()})
    ts = [d[0] for d in data]
    out.index = pd.to_datetime(ts, format="%Y/%m/%d %H:%M:%S")
    return out.apply(pd.to_numeric, errors="coerce")


def load_jma() -> pd.DataFrame:
    z = zipfile.ZipFile(_latest("jma_hourly_kanto8_*.zip"))
    parts = [_parse_jma_csv(z.read(n).decode("cp932")) for n in sorted(z.namelist())]
    w = pd.concat(parts).sort_index()
    w = w[~w.index.duplicated(keep="first")]
    full = pd.date_range(w.index.min(), w.index.max(), freq="1h")
    w = w.reindex(full)
    # 短い欠測は線形補間（気温・湿度）、日照は 0 埋め
    tcols = [c for c in w.columns if c.startswith(("temp_", "rh_"))]
    w[tcols] = w[tcols].interpolate(limit=6, limit_direction="both")
    scols = [c for c in w.columns if c.startswith("sun_")]
    w[scols] = w[scols].fillna(0.0)
    return w


def align_weather_to_demand_hours(w: pd.DataFrame) -> pd.DataFrame:
    """観測時刻ベースの気象値を「需要の 1 時間区間 [h, h+1)」ベースに変換する。"""
    nxt = w.shift(-1)  # h+1 時の観測値を h 行へ
    out = pd.DataFrame(index=w.index)
    for c in w.columns:
        if c.startswith(("temp_", "rh_")):
            out[c] = (w[c] + nxt[c]) / 2.0
        elif c.startswith("sun_"):
            out[c] = nxt[c]
    return out


def add_area_weighted(w: pd.DataFrame) -> pd.DataFrame:
    """都県人口で重み付けした関東平均値を作る（欠測地点は除いて重みを再正規化）。"""
    weights = {suf: pop for _, (suf, pop) in STATIONS.items()}
    for key in ["temp", "rh", "sun"]:
        vals = pd.DataFrame({suf: w[f"{key}_{suf}"] for suf in weights})
        wts = pd.DataFrame({suf: np.where(vals[suf].notna(), pop, 0.0) for suf, pop in weights.items()},
                           index=vals.index)
        num = (vals.fillna(0.0) * wts).sum(axis=1)
        den = wts.sum(axis=1).replace(0.0, np.nan)
        w[f"{key}_kanto"] = (num / den).interpolate(limit_direction="both")
    return w


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tepco = load_tepco()
    jma = load_jma()
    jma = add_area_weighted(align_weather_to_demand_hours(jma))
    df = tepco.join(jma, how="left")
    df.index.name = "datetime"

    # 品質チェック
    n_missing_demand = int(df["demand"].isna().sum())
    n_missing_temp = int(df["temp_kanto"].isna().sum())
    gap = (df["demand"] - df["demand_from_5min"]).abs()
    print(f"期間: {df.index.min()} 〜 {df.index.max()}  行数: {len(df):,}")
    print(f"需要欠測: {n_missing_demand}  気温欠測: {n_missing_temp}")
    print(f"1時間値と5分値平均の差: 中央値 {gap.median():.1f} 万kW / 95%点 {gap.quantile(0.95):.1f} 万kW")
    print(df[["demand", "tepco_forecast", "solar", "temp_kanto", "rh_kanto", "sun_kanto"]].describe().round(2))

    df.to_csv(OUT / "hourly_dataset.csv", float_format="%.3f")
    print("saved:", OUT / "hourly_dataset.csv")


if __name__ == "__main__":
    main()
