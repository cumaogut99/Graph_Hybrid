@echo off
REM Smart Downsampler Build Script
REM ================================
REM Bu script C++ modülünü derler ve projenin kök dizinine kopyalar.

echo.
echo ============================================
echo   Smart Downsampler Build Script
echo   MachinePulseAI - High Performance DSP
echo ============================================
echo.

cd /d "%~dp0"

REM Build klasörü yoksa oluştur
if not exist "build" mkdir build

cd build

echo [1/3] Configuring CMake...
cmake .. -G "Visual Studio 17 2022" -A x64
if errorlevel 1 (
    echo CMake configuration failed!
    pause
    exit /b 1
)

echo.
echo [2/3] Building Release...
cmake --build . --config Release --parallel
if errorlevel 1 (
    echo Build failed!
    pause
    exit /b 1
)

echo.
echo [3/3] Installing to project root...
cmake --install . --config Release
if errorlevel 1 (
    echo Install failed!
    pause
    exit /b 1
)

echo.
echo ============================================
echo   Build completed successfully!
echo   Module: time_graph_cpp.pyd
echo ============================================
echo.

REM Test import
echo Testing module import...
cd ..\..
python -c "import time_graph_cpp as tg; print(f'Module loaded: {dir(tg)}')"
if errorlevel 1 (
    echo Module import test failed!
) else (
    echo Module import test passed!
)

echo.
pause
