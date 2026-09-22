# Daily paper-trading run: download prices, fill orders, maybe re-learn, queue new orders.
# Intended to run after the US close (16:30 New York time or later); output is appended to logs/.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force -Path "$root\logs" | Out-Null
$log = "$root\logs\paper-$(Get-Date -Format 'yyyy-MM').log"
"==== $(Get-Date -Format 's') ====" | Out-File -Append -Encoding utf8 $log
uv run evotrader paper run *>&1 | Out-File -Append -Encoding utf8 $log
