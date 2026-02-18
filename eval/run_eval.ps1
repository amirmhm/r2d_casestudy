# =============================================================
# GGF LLM Systems Case — Evaluation Runner (Windows PowerShell)
# =============================================================
# Usage: .\eval\run_eval.ps1
# =============================================================

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$RepoRoot = Split-Path -Parent $ScriptDir

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host " GGF LLM Systems Case - Evaluation Runner" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check .env
$EnvFile = Join-Path $RepoRoot ".env"
$EnvExample = Join-Path $RepoRoot ".env.example"

if (-not (Test-Path $EnvFile)) {
    Write-Host "[WARN] No .env file found. Copying .env.example..." -ForegroundColor Yellow
    Copy-Item $EnvExample $EnvFile
    Write-Host "[WARN] Please edit .env with your actual API key before running." -ForegroundColor Yellow
    exit 1
}

# Load .env
Get-Content $EnvFile | ForEach-Object {
    if ($_ -match "^\s*([^#][^=]+)=(.*)$") {
        $key = $Matches[1].Trim()
        $value = $Matches[2].Trim()
        [System.Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

# Check API key
$ApiKey = [System.Environment]::GetEnvironmentVariable("OPENAI_API_KEY", "Process")
if ([string]::IsNullOrEmpty($ApiKey) -or $ApiKey -eq "YOUR_KEY_HERE") {
    Write-Host "[ERROR] OPENAI_API_KEY is not set. Please update .env" -ForegroundColor Red
    exit 1
}

# Check Node
try {
    $nodeVersion = node --version
    Write-Host "[OK] Node.js $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Node.js is required but not found." -ForegroundColor Red
    exit 1
}

# Check Python
$pythonCmd = $null
try {
    $pyVersion = python --version 2>&1
    $pythonCmd = "python"
    Write-Host "[OK] $pyVersion" -ForegroundColor Green
} catch {
    try {
        $pyVersion = python3 --version 2>&1
        $pythonCmd = "python3"
        Write-Host "[OK] $pyVersion" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Python 3.11+ is required but not found." -ForegroundColor Red
        exit 1
    }
}

# Install Node dependencies
Write-Host ""
Write-Host "[STEP 1] Installing Node dependencies..." -ForegroundColor Blue
Push-Location (Join-Path $RepoRoot "ggf-mini-game")
npm install --silent
Pop-Location

# Build baseline
Write-Host ""
Write-Host "[STEP 2] Building baseline..." -ForegroundColor Blue
Push-Location (Join-Path $RepoRoot "ggf-mini-game")
npm run build
Pop-Location
Write-Host "[OK] Baseline build succeeded" -ForegroundColor Green

# Run baseline sanity
Write-Host ""
Write-Host "[STEP 3] Running baseline sanity check..." -ForegroundColor Blue
Push-Location $RepoRoot
node eval/checks/baseline_sanity.mjs
Pop-Location

# Install Python dependencies
Write-Host ""
Write-Host "[STEP 4] Installing Python dependencies..." -ForegroundColor Blue
$SolutionDir = Join-Path $RepoRoot "solution"
Push-Location $SolutionDir

$VenvDir = Join-Path $SolutionDir ".venv"
if (-not (Test-Path $VenvDir)) {
    & $pythonCmd -m venv .venv
}

$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
& $ActivateScript
pip install -e . --quiet

Pop-Location

# Run evaluation
Write-Host ""
Write-Host "[STEP 5] Running evaluation..." -ForegroundColor Blue
Push-Location $RepoRoot
& $pythonCmd -m ggf_case.cli run-eval --output-dir eval/outputs
Pop-Location

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host " Evaluation complete!" -ForegroundColor Green
Write-Host " Results: eval/outputs/" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
