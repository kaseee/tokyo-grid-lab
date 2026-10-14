"""
アプリ同梱アセットの生成
  - Noto Sans CJK JP のサブセット（アプリで使う文字＋かな・英数記号のみ）
  - アプリアイコン（.ico / .png）とスプラッシュ画像
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from fontTools import subset
from fontTools.ttLib import TTCollection
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app"
OUT = APP / "assets"
NOTO = Path("/usr/share/fonts/opentype/noto")


def collect_chars() -> str:
    chars = set()
    for p in list(APP.rglob("*.py")) + [OUT / "metrics.json"]:
        chars |= set(p.read_text(encoding="utf-8"))
    # かな・英数・記号を一通り
    for a, b in [(0x20, 0x7E), (0x3000, 0x30FF), (0xFF01, 0xFF5E), (0x2010, 0x2027), (0x2190, 0x21FF),
                 (0x2460, 0x2473), (0x25A0, 0x25FF), (0x00B0, 0x00B7), (0x00D7, 0x00D7), (0x2212, 0x2212),
                 (0x2103, 0x2103), (0x00B2, 0x00B3)]:
        chars |= {chr(c) for c in range(a, b + 1)}
    # よく使う漢字（曜日・月・数量・単位）
    chars |= set("日月火水木金土年時分秒週間曜祝休平前後今昨翌当初終最大小高低円万億千百十一二三四五六七八九〇")
    return "".join(sorted(c for c in chars if c.isprintable() and c not in "\n\r\t"))


def make_fonts(text: str) -> None:
    (OUT / "fonts").mkdir(parents=True, exist_ok=True)
    for weight in ("Regular", "Bold"):
        src = NOTO / f"NotoSansCJK-{weight}.ttc"
        coll = TTCollection(str(src))
        font = coll.fonts[0]  # 0 = Noto Sans CJK JP
        tmp = OUT / "fonts" / f"_tmp_{weight}.otf"
        font.save(str(tmp))
        opts = subset.Options()
        opts.layout_features = ["*"]
        opts.name_IDs = ["*"]
        opts.name_languages = ["*"]
        opts.notdef_outline = True
        opts.glyph_names = False
        opts.hinting = True
        f = subset.load_font(str(tmp), opts)
        sub = subset.Subsetter(opts)
        sub.populate(text=text)
        sub.subset(f)
        out = OUT / "fonts" / f"NotoSansCJKjp-{weight}-subset.otf"
        subset.save_font(f, str(out), opts)
        tmp.unlink()
        print(out.name, f"{out.stat().st_size / 1024:.0f} KB")
    lic = Path("/usr/share/doc/fonts-noto-cjk/copyright")
    (OUT / "fonts" / "LICENSE_Noto_Sans_CJK.txt").write_text(
        "Noto Sans CJK JP (subset) - Copyright 2014-2021 Adobe (http://www.adobe.com/), "
        "with Reserved Font Name 'Source'.\nLicensed under the SIL Open Font License, Version 1.1. "
        "https://openfontlicense.org\n\n" + (lic.read_text(encoding="utf-8", errors="ignore")[:4000] if lic.exists() else ""),
        encoding="utf-8")


def _font(size, bold=False):
    return ImageFont.truetype(str(NOTO / ("NotoSansCJK-Bold.ttc" if bold else "NotoSansCJK-Regular.ttc")), size, index=0)


def make_icon() -> None:
    S = 256
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((8, 8, S - 8, S - 8), radius=52, fill=(29, 32, 38, 255))
    bars = [((58, 150, 96, 206), (98, 80, 214)), ((110, 104, 148, 206), (42, 120, 214)), ((162, 56, 200, 206), (237, 161, 0))]
    for box, col in bars:
        d.rounded_rectangle(box, radius=8, fill=col + (255,))
    d.line([(52, 128), (104, 92), (150, 112), (206, 52)], fill=(255, 255, 255, 235), width=10, joint="curve")
    img.save(OUT / "app_icon.png")
    img.save(OUT / "app.ico", sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    print("icon ok")


def make_splash() -> None:
    W, H = 560, 300
    img = Image.new("RGB", (W, H), (29, 32, 38))
    d = ImageDraw.Draw(img)
    icon = Image.open(OUT / "app_icon.png").resize((72, 72), Image.LANCZOS)
    img.paste(icon, (40, 46), icon)
    d.text((132, 50), "Tokyo Grid Lab", font=_font(30, True), fill=(255, 255, 255))
    d.text((134, 94), "東京エリア 電力需要予測 × 発電計画最適化", font=_font(17), fill=(201, 204, 211))
    d.text((40, 170), "LightGBM による前日需要予測（MAPE 評価）", font=_font(15), fill=(160, 165, 175))
    d.text((40, 196), "PuLP ＋ HiGHS による起動停止つき発電計画（MILP）", font=_font(15), fill=(160, 165, 175))
    d.rectangle((40, 236, W - 40, 237), fill=(50, 56, 68))
    d.text((40, 250), "起動中…（初回は展開に数十秒かかることがあります）", font=_font(13), fill=(125, 130, 141))
    img.save(OUT / "splash.png")
    print("splash ok")


if __name__ == "__main__":
    text = collect_chars()
    print("chars:", len(text))
    make_fonts(text)
    make_icon()
    make_splash()
