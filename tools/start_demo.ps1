# Start the RoadSight live demo on this machine and publish it through a Cloudflare quick tunnel.
#
#   powershell -ExecutionPolicy Bypass -File tools\start_demo.ps1
#
# Starts the FastAPI backend (web/app.py, same code and weights as the submission) on port 7860, opens a quick
# tunnel (no account needed; the address changes every start), and, when build\static-site is the git clone of
# the Hugging Face Static Space, writes the address to data/api.json there and pushes it, so the website's Demo
# page uses this machine. Keep the window open while the demo should be reachable; Ctrl+C stops everything.

$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv-local\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$cf = Join-Path $root "build\cloudflared.exe"
if (-not (Test-Path $cf)) {
    Write-Host "Downloading cloudflared..."
    Invoke-WebRequest "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile $cf -UseBasicParsing
}
$logs = Join-Path $root "outputs"
New-Item -ItemType Directory -Force $logs | Out-Null
$serverLog = Join-Path $logs "demo_server.log"
$tunnelLog = Join-Path $logs "tunnel.log"
Remove-Item $tunnelLog, "$tunnelLog.out" -ErrorAction SilentlyContinue

$server = Start-Process $py -ArgumentList "-m", "uvicorn", "web.app:app", "--host", "127.0.0.1", "--port", "7860" `
    -WorkingDirectory $root -RedirectStandardError $serverLog -RedirectStandardOutput "$serverLog.out" -PassThru -WindowStyle Hidden
$tunnel = Start-Process $cf -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://localhost:7860" `
    -RedirectStandardError $tunnelLog -RedirectStandardOutput "$tunnelLog.out" -PassThru -WindowStyle Hidden

try {
    $url = $null
    for ($i = 0; $i -lt 90 -and -not $url; $i++) {
        Start-Sleep 2
        if (Test-Path $tunnelLog) {
            $m = Select-String -Path $tunnelLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
            if ($m) { $url = $m.Matches[0].Value }
        }
    }
    if (-not $url) { throw "The tunnel did not start; see $tunnelLog" }
    Write-Host "Tunnel: $url (waiting for the models to load...)"

    $ok = $false
    for ($i = 0; $i -lt 90 -and -not $ok; $i++) {
        Start-Sleep 2
        try { $h = Invoke-RestMethod "$url/api/health" -TimeoutSec 10; $ok = [bool]$h.ok } catch { }
    }
    if (-not $ok) { throw "The demo server did not answer; see $serverLog" }

    $site = Join-Path $root "build\static-site"
    if (Test-Path (Join-Path $site ".git")) {
        Set-Content -Path (Join-Path $site "data\api.json") -Value "{`n  `"api`": `"$url`"`n}" -Encoding ascii
        git -C $site add data/api.json
        git -C $site commit -q -m "Demo backend: $url"
        git -C $site push -q
        if ($LASTEXITCODE -eq 0) { Write-Host "The website's Demo page now uses this machine (the Space updates within a minute)." }
        else { Write-Host "Could not push data/api.json; set it on the Space by hand to: $url" }
    } else {
        Write-Host "build\static-site is not the Space clone yet: set data/api.json on the Space by hand to: $url"
    }

    Write-Host ""
    Write-Host "LIVE DEMO RUNNING at $url"
    Write-Host "Keep this window open (and the laptop awake and online). Press Ctrl+C to stop."
    while (-not $server.HasExited -and -not $tunnel.HasExited) { Start-Sleep 5 }
    Write-Host "The server or the tunnel stopped; see $serverLog and $tunnelLog"
} finally {
    foreach ($p in @($server, $tunnel)) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force } }
}
