# Stop execution immediately if any command fails
$ErrorActionPreference = "Stop"

# Helper to enforce strict stopping on native CLI failures
function Exec-Native {
    param([scriptblock]$ScriptBlock)
    & $ScriptBlock
    if ($LASTEXITCODE -ne 0) {
        Write-Host "`n[!] Error: Command failed with exit code $LASTEXITCODE. Aborting launch!" -ForegroundColor Red
        Exit $LASTEXITCODE
    }
}

Write-Host " Launching JupyterLab..." -ForegroundColor Cyan

# Enforce UTF-8 for Python I/O
$env:PYTHONUTF8 = "1"

# Flag session as rpy2 to prevent .Rprofile boot loops
$env:RPY2_SESSION = "1"
Write-Host "`n[1/4] Set process-local PYTHONUTF8 and RPY2_SESSION" -ForegroundColor Gray

# Query R 4.4.1 path directly
try {
    $rHomePath = (rig run 4.4.1 Rscript -e "cat(R.home())")
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($rHomePath)) {
        throw "Failed to query R 4.4.1 path."
    }
    Write-Host "[2/4] Discovered R 4.4.1 -> $rHomePath" -ForegroundColor Gray
} catch {
    Write-Host "[!] Error: Could not locate R 4.4.1 via rig. Ensure R 4.4.1 is installed." -ForegroundColor Red
    Exit 1
}

# Set process-scoped R_HOME and PATH for rpy2
$env:R_HOME = $rHomePath
$env:PATH = "$rHomePath\bin\x64;$rHomePath\bin;$env:PATH"
Write-Host "[3/4] Injected local R_HOME and PATH into session" -ForegroundColor Gray

# Restore R dependencies if missing & Launch Jupyter
Write-Host "[4/4] Restoring R packages and starting JupyterLab...`n" -ForegroundColor Green
Exec-Native { rig run 4.4.1 Rscript -e "renv::restore()" }
Exec-Native { uv run jupyter lab }
