# -*- mode: python ; coding: utf-8 -*-
# PyInstaller 用ビルド設定（Windows 単一 exe）
#   python -m PyInstaller --noconfirm --clean build/TokyoGridLab.spec
import os
import sys

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
APP = os.path.join(ROOT, "app")
ASSETS = os.path.join(APP, "assets")

datas = [
    (os.path.join(ASSETS, "app_data.npz"), "assets"),
    (os.path.join(ASSETS, "metrics.json"), "assets"),
    (os.path.join(ASSETS, "app.ico"), "assets"),
    (os.path.join(ASSETS, "app_icon.png"), "assets"),
    (os.path.join(ASSETS, "fonts"), os.path.join("assets", "fonts")),
]

# LightGBM（vcomp140）と HiGHS（msvcp140）が使う MSVC ランタイムを同梱（未インストールの PC 向け）
binaries = []
for dll in ("vcruntime140.dll", "vcruntime140_1.dll", "vcomp140.dll", "msvcp140.dll", "msvcp140_1.dll"):
    for base in (sys.prefix, sys.base_prefix):  # venv の場合は msvc-runtime が sys.prefix 側に入る
        p = os.path.join(base, dll)
        if os.path.exists(p):
            binaries.append((p, "."))
            break

a = Analysis(
    [os.path.join(APP, "tepco_app.py")],
    pathex=[APP],
    binaries=binaries,
    datas=datas,
    hiddenimports=["highspy", "lightgbm", "pulp.apis.highs_api", "matplotlib.backends.backend_tkagg"],
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["TkAgg"]}},
    runtime_hooks=[],
    excludes=["pandas", "pyarrow", "dask", "sklearn", "IPython", "jupyter", "notebook", "PyQt5", "PyQt6",
              "PySide2", "PySide6", "wx", "gi", "pulp.tests", "numpy.tests", "scipy.tests", "tkinter.test",
              "pydoc_data", "lib2to3", "setuptools", "pkg_resources",
              # LightGBM が使うのは scipy.sparse / scipy.linalg だけなので、他のサブパッケージは同梱しない
              "scipy.cluster", "scipy.constants", "scipy.datasets", "scipy.fft", "scipy.fftpack",
              "scipy.integrate", "scipy.interpolate", "scipy.io", "scipy.misc", "scipy.ndimage", "scipy.odr",
              "scipy.optimize", "scipy.signal", "scipy.spatial", "scipy.special", "scipy.stats",
              # Pillow の未使用プラグイン
              "PIL._avif", "PIL.AvifImagePlugin", "PIL._webp", "PIL.WebPImagePlugin", "PIL._imagingcms",
              "PIL.ImageCms", "PIL.ImageQt"],
    noarchive=False,
    optimize=1,
)

# PuLP 同梱の CBC 実行ファイルは使わない（ソルバーは HiGHS）ので除外してサイズを削減
def _keep(entry):
    dest = entry[0].replace("\\", "/")
    return "pulp/solverdir" not in dest

a.datas = [d for d in a.datas if _keep(d)]
a.binaries = [b for b in a.binaries if _keep(b)]

pyz = PYZ(a.pure)

splash = Splash(
    os.path.join(ASSETS, "splash.png"),
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    minify_script=True,
    always_on_top=False,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    splash,
    splash.binaries,
    [],
    name="TokyoGridLab",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    icon=os.path.join(ASSETS, "app.ico"),
    version=os.path.join(SPECPATH, "version_info.txt"),
)
