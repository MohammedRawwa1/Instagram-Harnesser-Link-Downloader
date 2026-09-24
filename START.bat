@echo off
title Instagram Reel Transcriber
cd /d "%~dp0"
.venv\Scripts\python.exe transcribe.py
pause
