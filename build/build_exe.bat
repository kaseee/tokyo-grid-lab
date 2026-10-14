@echo off
rem Build TokyoGridLab.exe on Windows (requires 64-bit Python 3.12)
cd /d %~dp0\..
if not exist .venv (
  python -m venv .venv
)
call .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt pyinstaller==6.22.3 msvc-runtime
python -m PyInstaller --noconfirm --clean --distpath dist --workpath build\work build\TokyoGridLab.spec
echo.
echo Done: dist\TokyoGridLab.exe
pause
