# Starts the examiner app: API server (port 8765, real checkpoints on CPU) and web UI (port 5180).
# Usage from the repo root:  powershell -File examiner_app\start.ps1
$ErrorActionPreference = 'Stop'
$app = $PSScriptRoot
$repo = Split-Path $app -Parent
$python = Join-Path $repo '.venv\Scripts\python.exe'
if (-not (Test-Path $python)) { throw "Python venv not found at $python" }
if (-not (Test-Path (Join-Path $app 'web\node_modules'))) { npm --prefix (Join-Path $app 'web') install }

$server = Start-Process -FilePath $python -ArgumentList '-m', 'uvicorn', 'app:app', '--port', '8765' `
  -WorkingDirectory (Join-Path $app 'server') -PassThru -NoNewWindow
try {
  Start-Sleep -Seconds 2
  Start-Process 'http://localhost:5180/'
  npm --prefix (Join-Path $app 'web') run dev
} finally {
  if (-not $server.HasExited) { Stop-Process -Id $server.Id }
}
