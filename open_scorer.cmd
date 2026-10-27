@echo off
cd /d "%~dp0"
py -3.13 -m spm_rating.gui
if errorlevel 1 pause
