# =============================================================================
# Smart B-Roll Inserter - Windows PowerShell Startup Script
# =============================================================================

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Smart B-Roll Inserter - Starting..."    -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# Get the script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# =============================================================================
# Check Python
# =============================================================================
Write-Host "Checking Python installation..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "Using: $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Python is not installed. Please install Python 3.8+" -ForegroundColor Red
    exit 1
}

# =============================================================================
# Check Node.js
# =============================================================================
Write-Host "Checking Node.js installation..." -ForegroundColor Yellow
try {
    $nodeVersion = node --version 2>&1
    Write-Host "Using: Node $nodeVersion" -ForegroundColor Green
} catch {
    Write-Host "Error: Node.js is not installed. Please install Node.js 16+" -ForegroundColor Red
    exit 1
}

# =============================================================================
# Setup Python Virtual Environment
# =============================================================================
Write-Host "Setting up Python virtual environment..." -ForegroundColor Yellow

if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# =============================================================================
# Install Backend Dependencies
# =============================================================================
Write-Host "Installing backend dependencies..." -ForegroundColor Yellow
Set-Location backend
pip install -r requirements.txt --quiet
Set-Location ..

# =============================================================================
# Install Frontend Dependencies
# =============================================================================
Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
Set-Location frontend
if (-not (Test-Path "node_modules")) {
    npm install
}
Set-Location ..

# =============================================================================
# Create .env file if not exists
# =============================================================================
if (-not (Test-Path "backend\.env")) {
    Write-Host "Creating .env file..." -ForegroundColor Yellow
    $envContent = @"
# API Provider: "openai", "gemini", "openrouter", or "offline"
API_PROVIDER=offline

# Google Gemini API Key (get from https://makersuite.google.com/app/apikey)
GEMINI_API_KEY=

# OpenAI API Key (optional)
OPENAI_API_KEY=

# OpenRouter API Key (optional - for access to multiple models)
OPENROUTER_API_KEY=
"@
    $envContent | Out-File -FilePath "backend\.env" -Encoding utf8
    Write-Host ".env file created. Edit backend\.env to add your API keys." -ForegroundColor Green
}

# =============================================================================
# Start Services
# =============================================================================
Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Starting Services..."                    -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Start Backend in new terminal
Write-Host "Starting Backend on http://localhost:8000" -ForegroundColor Green
$backendCmd = "cd '$ScriptDir\backend'; & '$ScriptDir\venv\Scripts\Activate.ps1'; python -m uvicorn app:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

# Wait for backend to start
Start-Sleep -Seconds 3

# Start Frontend in new terminal
Write-Host "Starting Frontend on http://localhost:3000" -ForegroundColor Green
$frontendCmd = "cd '$ScriptDir\frontend'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Services Running!"                       -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Backend API:  " -NoNewline; Write-Host "http://localhost:8000" -ForegroundColor Cyan
Write-Host "  Frontend UI:  " -NoNewline; Write-Host "http://localhost:3000" -ForegroundColor Cyan
Write-Host "  API Docs:     " -NoNewline; Write-Host "http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host ""
Write-Host "Close the terminal windows to stop the services." -ForegroundColor Yellow
Write-Host ""

# Open browser after a short delay
Start-Sleep -Seconds 2
Start-Process "http://localhost:3001"
