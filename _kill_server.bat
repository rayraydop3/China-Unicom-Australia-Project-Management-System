@echo off
taskkill /F /IM python.exe /FI "WINDOWTITLE eq *server*" >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":5000" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)
