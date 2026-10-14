"""
公開データのダウンロード（data_raw/ に保存）

  1. 東京電力パワーグリッド「でんき予報」過去の電力使用実績データ（月別 zip、2022年4月以降の形式）
     https://www.tepco.co.jp/forecast/html/download-j.html
  2. 気象庁「過去の気象データ・ダウンロード」時別値（関東 8 地点：気温・相対湿度・日照時間）
     https://www.data.jma.go.jp/risk/obsdl/index.php

使い方:
    python tools/download_data.py                       # 2022-04-01 〜 昨日
    python tools/download_data.py --start 2022-04 --end 2026-10-12

出力（src/prepare_data.py がそのまま読み込む形式）:
    data_raw/tepco_power_usage_YYYYMM-YYYYMM.zip      月別 zip をまとめた zip
    data_raw/jma_hourly_kanto8_YYYYMM-YYYYMM.zip      月ごとの CSV（Shift_JIS）をまとめた zip
                                                      ※ 気象は特徴量計算のため 1 か月前から取得

サーバーに負荷をかけないよう、1 リクエストごとに待ち時間を入れています。
"""
from __future__ import annotations

import argparse
import calendar
import datetime as dt
import io
import json
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_raw"

TEPCO_URL = "https://www.tepco.co.jp/forecast/html/images/{ym}_power_usage.zip"
JMA_PAGE = "https://www.data.jma.go.jp/risk/obsdl/index.php"
JMA_TABLE = "https://www.data.jma.go.jp/risk/obsdl/show/table"

# 気象官署：東京・横浜・千葉・熊谷・水戸・宇都宮・前橋・甲府
JMA_STATIONS = ["s47662", "s47670", "s47682", "s47626", "s47629", "s47615", "s47624", "s47638"]
# 要素：気温(201)・相対湿度(605)・日照時間(401)、時別値は aggrgPeriod=9
JMA_ELEMENTS = [["201", ""], ["605", ""], ["401", ""]]

HEADERS = {"User-Agent": "Mozilla/5.0 (tokyo-grid-lab data downloader)"}


def month_iter(start: dt.date, end: dt.date):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def fetch(req: urllib.request.Request, retries: int = 3, wait: float = 3.0) -> tuple[bytes, dict]:
    for k in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.read(), dict(r.headers)
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                raise
            print(f"  retry ({e})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def download_tepco(start: dt.date, end: dt.date, sleep: float) -> Path:
    out = RAW / f"tepco_power_usage_{start:%Y%m}-{end:%Y%m}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:
        for y, m in month_iter(start, end):
            ym = f"{y}{m:02d}"
            data, _ = fetch(urllib.request.Request(TEPCO_URL.format(ym=ym), headers=HEADERS))
            if not data.startswith(b"PK"):
                raise RuntimeError(f"{ym}: zip ではない応答です")
            z.writestr(f"{ym}_power_usage.zip", data)
            print(f"TEPCO {ym}: {len(data) / 1024:.0f} KB")
            time.sleep(sleep)
    return out


def download_jma(start: dt.date, end: dt.date, sleep: float) -> Path:
    out = RAW / f"jma_hourly_kanto8_{start:%Y%m}-{end:%Y%m}.zip"
    hdr = dict(HEADERS, Referer=JMA_PAGE, Origin="https://www.data.jma.go.jp",
               **{"Content-Type": "application/x-www-form-urlencoded"})
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for y, m in month_iter(start, end):
            last = end.day if (y, m) == (end.year, end.month) else calendar.monthrange(y, m)[1]
            params = {
                "stationNumList": json.dumps(JMA_STATIONS),
                "aggrgPeriod": "9",
                "elementNumList": json.dumps(JMA_ELEMENTS),
                "interAnnualType": "1",
                "ymdList": json.dumps([str(y), str(y), str(m), str(m), "1", str(last)]),
                "option": json.dumps({"list": ["obs"], "nyear": 0}),
                "downloadFlag": "true", "rmkFlag": "1", "disconnectFlag": "1", "youbiFlag": "0",
                "fukenFlag": "0", "kijiFlag": "0", "csvFlag": "1", "jikantaiFlag": "0",
                "jikantaiList": "[1,24]", "ymdLiteral": "1",
            }
            req = urllib.request.Request(JMA_TABLE, data=urllib.parse.urlencode(params).encode(), headers=hdr)
            data, headers = fetch(req)
            disp = next((v for k, v in headers.items() if k.lower() == "content-disposition"), "")
            if not disp.startswith("attachment"):
                raise RuntimeError(f"{y}-{m:02d}: CSV がダウンロードできませんでした（データ量の上限などを確認）")
            z.writestr(f"jma_hourly_kanto8_{y}{m:02d}.csv", data)
            print(f"JMA   {y}-{m:02d}: {len(data) / 1024:.0f} KB")
            time.sleep(sleep)
    return out


def parse_date(s: str, end: bool) -> dt.date:
    if len(s) == 7:  # YYYY-MM
        y, m = map(int, s.split("-"))
        return dt.date(y, m, calendar.monthrange(y, m)[1] if end else 1)
    return dt.date.fromisoformat(s)


def main() -> None:
    ap = argparse.ArgumentParser(description="でんき予報・気象庁の公開データをダウンロード")
    ap.add_argument("--start", default="2022-04", help="開始（YYYY-MM または YYYY-MM-DD）")
    ap.add_argument("--end", default=(dt.date.today() - dt.timedelta(days=1)).isoformat(),
                    help="終了（YYYY-MM または YYYY-MM-DD、既定は昨日）")
    ap.add_argument("--sleep", type=float, default=1.0, help="リクエスト間隔（秒）")
    a = ap.parse_args()
    start, end = parse_date(a.start, False), parse_date(a.end, True)
    RAW.mkdir(parents=True, exist_ok=True)
    print("saved:", download_tepco(start, end, a.sleep))
    wstart = (start.replace(day=1) - dt.timedelta(days=1)).replace(day=1)  # 気象は 1 か月前から
    print("saved:", download_jma(wstart, end, a.sleep))


if __name__ == "__main__":
    main()
