@echo off
echo ============================================================
echo Quantum Chemical Calculations Step Maker - Web Server
echo ============================================================
echo.

REM Get IP address
python get_ip.py
echo.

echo Starting web server...
echo Press Ctrl+C to stop the server
echo.
echo ============================================================

python quantum_steps_web.py


