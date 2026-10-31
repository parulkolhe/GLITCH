# GLITCH: one-command start (Windows PowerShell):  powershell -ExecutionPolicy Bypass -File play.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
if (-not (Test-Path ".venv/Scripts/python.exe")) {
  Write-Host "First run: creating .venv and installing the game server (no TensorFlow needed)..."
  python -m venv .venv
  .venv/Scripts/python.exe -m pip install -r backend/requirements.txt
}
$port = if ($env:GLITCH_PORT) { $env:GLITCH_PORT } else { "8001" }
Start-Process "http://127.0.0.1:$port"
.venv/Scripts/python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port $port
