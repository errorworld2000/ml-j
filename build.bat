@echo off
REM 判断操作系统类型
ver | findstr /i "windows" >nul
if %errorlevel% == 0 (
    echo Windows 系统 detected.
    python3 -m build
) else (
    echo Unix/Linux/macOS 系统 detected.
    python3 -m build
)
