@echo off
REM Build script for Windows (MSVC)
REM
REM Prerequisites:
REM   - Visual Studio 2019 or later
REM   - Qt5 installed
REM   - Python 3.8+ with development headers
REM
REM Usage:
REM   build.cmd [Release|Debug]

setlocal

set BUILD_TYPE=%1
if "%BUILD_TYPE%"=="" set BUILD_TYPE=Release

echo ================================================
echo Time Graph C++ Build Script (Windows)
echo ================================================
echo.
echo Build Type: %BUILD_TYPE%
echo.

REM Detect Qt5 installation
set QT5_DIR=
if exist "C:\Qt\5.15.2\msvc2019_64\lib\cmake\Qt5" (
    set QT5_DIR=C:\Qt\5.15.2\msvc2019_64\lib\cmake\Qt5
    echo Qt5 found: %QT5_DIR%
) else (
    echo WARNING: Qt5 not found in default location!
    echo Please set QT5_DIR manually or install Qt5 to C:\Qt\5.15.2\msvc2019_64
    echo.
)

REM Configure
echo.
echo [1/3] Configuring CMake...
echo ================================================

if not "%QT5_DIR%"=="" (
    cmake -B build -G "Visual Studio 17 2022" -A x64 ^
          -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
          -DQt5_DIR=%QT5_DIR%
) else (
    cmake -B build -G "Visual Studio 17 2022" -A x64 ^
          -DCMAKE_BUILD_TYPE=%BUILD_TYPE%
)

if errorlevel 1 (
    echo.
    echo ❌ CMake configuration failed!
    echo.
    echo Troubleshooting:
    echo   1. Make sure Qt5 is installed
    echo   2. Set Qt5_DIR manually: set Qt5_DIR=C:\path\to\qt5\lib\cmake\Qt5
    echo   3. Check CMake output for errors
    exit /b 1
)

REM Build
echo.
echo [2/3] Building...
echo ================================================
cmake --build build --config %BUILD_TYPE% --parallel

if errorlevel 1 (
    echo.
    echo ❌ Build failed!
    exit /b 1
)

REM Install (copy .pyd to parent directory)
echo.
echo [3/3] Installing...
echo ================================================
cmake --install build --config %BUILD_TYPE%

if errorlevel 1 (
    echo.
    echo ❌ Install failed!
    exit /b 1
)

echo.
echo ================================================
echo ✅ Build completed successfully!
echo ================================================
echo.
echo Module location: time_graph_cpp.pyd (in parent directory)
echo.
echo To test:
echo   python test_build.py
echo.

endlocal

