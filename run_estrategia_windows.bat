@echo off
setlocal enabledelayedexpansion

REM Uso:
REM   run_estrategia_windows.bat [args de estrategia.py]

set VENV_DIR=.venv_estrategia

if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo [INFO] Creando entorno virtual en %VENV_DIR%...
  py -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual.
    exit /b 1
  )
)

echo [INFO] Activando entorno virtual...
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
  echo [ERROR] No se pudo activar el entorno virtual.
  exit /b 1
)

echo [INFO] Actualizando pip...
python -m pip install --upgrade pip >nul

echo [INFO] Instalando dependencias (python-binance, matplotlib, pandas, openpyxl)...
pip install -q python-binance matplotlib pandas openpyxl
if errorlevel 1 (
  echo [WARN] Fallo parcial instalando dependencias. Intentando ejecutar igual...
)

echo [INFO] Ejecutando visualizador...
python estrategia.py %*
set EXIT_CODE=%ERRORLEVEL%

echo.
echo [INFO] Proceso finalizado con codigo %EXIT_CODE%.
exit /b %EXIT_CODE%
