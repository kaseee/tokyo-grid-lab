"""特徴量のメタ情報（名前・日本語ラベル・グループ）。学習スクリプトとアプリで共通利用する。"""
from __future__ import annotations

FEATURE_GROUPS: dict[str, list[str]] = {
    "calendar": [
        "hour", "dow", "month", "doy_sin", "doy_cos",
        "is_holiday", "day_type", "is_obon", "is_newyear", "is_gw", "is_bridge",
        "day_type_prev", "day_type_next",
    ],
    "weather": [
        "temp_kanto", "temp_tokyo", "rh_kanto", "sun_kanto", "thi",
        "temp_day_max", "temp_day_min", "temp_day_mean", "temp_prev_day_mean",
        "temp_roll3", "sun_day_total", "cdd", "hdd",
    ],
    "lag": [
        "lag48", "lag168", "day_mean_lag2", "day_max_lag2", "day_type_lag2", "day_type_lag7",
    ],
}

GROUP_LABELS_JA = {"calendar": "カレンダー", "weather": "気象", "lag": "需要ラグ"}

FEATURE_LABELS_JA = {
    "hour": "時刻", "dow": "曜日", "month": "月", "doy_sin": "年周期(sin)", "doy_cos": "年周期(cos)",
    "is_holiday": "祝日", "day_type": "平日/土/日祝", "is_obon": "お盆", "is_newyear": "年末年始",
    "is_gw": "GW", "is_bridge": "飛び石平日", "day_type_prev": "前日の日種別", "day_type_next": "翌日の日種別",
    "temp_kanto": "気温(関東加重)", "temp_tokyo": "気温(東京)", "rh_kanto": "湿度(関東加重)",
    "sun_kanto": "日照時間(関東加重)", "thi": "不快指数", "temp_day_max": "日最高気温",
    "temp_day_min": "日最低気温", "temp_day_mean": "日平均気温", "temp_prev_day_mean": "前日平均気温",
    "temp_roll3": "気温3時間平均", "sun_day_total": "日照時間(日合計)", "cdd": "冷房度時(>22℃)",
    "hdd": "暖房度時(<14℃)", "lag48": "2日前同時刻需要", "lag168": "7日前同時刻需要",
    "day_mean_lag2": "2日前平均需要", "day_max_lag2": "2日前最大需要",
    "day_type_lag2": "2日前の日種別", "day_type_lag7": "7日前の日種別",
}

ALL_FEATURES = [f for g in FEATURE_GROUPS.values() for f in g]

GROUP_OF = {f: g for g, fs in FEATURE_GROUPS.items() for f in fs}
LABEL_OF = FEATURE_LABELS_JA
