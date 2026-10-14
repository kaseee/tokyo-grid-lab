#!/bin/bash
# Linux 上で Windows 用 exe をビルドする手順（この作品の exe はこの方法で作成）
#   1) Wine 9 と Windows 版 CPython 3.12（python-build-standalone の install_only 版）を用意
#   2) win_amd64 のホイールを pip download で取得し、Wine の Python にオフラインでインストール
#   3) Wine 上で PyInstaller を実行
# ※ Wine 9.0 の ucrtbase は crealf 未実装のため NumPy 2.x は動かない → NumPy 1.26 系を使用
set -e
export WINEPREFIX=${WINEPREFIX:-$HOME/.wine64} WINEARCH=win64 WINEDEBUG=-all WINEDLLOVERRIDES="mscoree,mshtml="
PY=${WINPY:-./winpython/python.exe}
WHEELS=${WHEELS:-./wheels}
pip download --only-binary=:all: --platform win_amd64 --python-version 3.12 --implementation cp \
  --abi cp312 --abi abi3 --abi none -d "$WHEELS" -r requirements.txt pyinstaller==6.22.3 \
  pyinstaller-hooks-contrib msvc-runtime pefile pywin32-ctypes
wine "$PY" -m pip install --no-index --find-links "$WHEELS" numpy==1.26.4 scipy==1.14.1 matplotlib==3.10.7 \
  contourpy==1.3.3 lightgbm==4.7.0 pulp==3.3.2 highspy==1.13.1 pillow pyinstaller==6.22.3 \
  pyinstaller-hooks-contrib msvc-runtime < /dev/null 2>&1 | cat
wine "$PY" -m PyInstaller --noconfirm --clean --distpath dist --workpath build/work build/TokyoGridLab.spec < /dev/null 2>&1 | cat
