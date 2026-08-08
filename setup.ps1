# Stop execution immediately if any command fails
$ErrorActionPreference = "Stop"

# Helper to enforce strict stopping on native CLI failures
function Exec-Native {
    param([scriptblock]$ScriptBlock)
    & $ScriptBlock
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[!] Error: Command failed with exit code $LASTEXITCODE. Aborting setup!" -ForegroundColor Red
        Exit $LASTEXITCODE
    }
}

Write-Host " Starting Project Environment Bootstrap..." -ForegroundColor Cyan

# Install local Python
Write-Host "`n[1/3] Installing Python 3.11.15..." -ForegroundColor Yellow
Exec-Native { uv python install 3.11.15 }

# Install local R version
Write-Host "[2/3] Installing R 4.4.1..." -ForegroundColor Yellow
Exec-Native { rig install 4.4.1 }

# Sync Python packages
Write-Host "[3/3] Syncing Python dependencies..." -ForegroundColor Yellow
Exec-Native { uv sync }

Write-Host " Setup complete!" -ForegroundColor Green
