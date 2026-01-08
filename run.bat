@echo off
REM =============================================================================
REM Smart B-Roll Inserter - Windows Startup Script (CMD)
REM =============================================================================

echo ==========================================
echo   Smart B-Roll Inserter - Starting...
echo ==========================================

REM Get the script directory
cd /d "%~dp0"

REM =============================================================================
REM Check Python
REM =============================================================================
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed. Please install Python 3.8+
    pause
    exit /b 1
)
python --version

REM =============================================================================
REM Check Node.js
REM =============================================================================
echo Checking Node.js installation...
node --version >nul 2>&1
if errorlevel 1 (
    echo Error: Node.js is not installed. Please install Node.js 16+
    pause
    exit /b 1
)
echo Node %errorlevel%
node --version

REM =============================================================================
REM Setup Python Virtual Environment
REM =============================================================================
echo Setting up Python virtual environment...

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM =============================================================================
REM Install Backend Dependencies
REM =============================================================================
echo Installing backend dependencies...
cd backend
pip install -r requirements.txt --quiet
cd ..

REM =============================================================================
REM Install Frontend Dependencies
REM =============================================================================
echo Installing frontend dependencies...
cd frontend
if not exist "node_modules" (
    call npm install
)
cd ..

REM =============================================================================
REM Create .env file if not exists
REM =============================================================================
if not exist "backend\.env" (
    echo Creating .env file...
    (
        echo # API Provider: "openai", "gemini", "openrouter", or "offline"
        echo API_PROVIDER=offline
        echo.
        echo # Google Gemini API Key ^(get from https://makersuite.google.com/app/apikey^)
        echo GEMINI_API_KEY=
        echo.
        echo # OpenAI API Key ^(optional^)
        echo OPENAI_API_KEY=
        echo.
        echo # OpenRouter API Key ^(optional - for access to multiple models^)
        echo OPENROUTER_API_KEY=
    ) > backend\.env
    echo .env file created. Edit backend\.env to add your API keys.
)

REM =============================================================================
REM Start Services
REM =============================================================================
echo.
echo ==========================================
echo   Starting Services...
echo ==========================================

echo Starting Backend on http://localhost:8000
start "Backend Server" cmd /k "cd /d %~dp0backend && call ..\venv\Scripts\activate.bat && python -m uvicorn app:app --reload --port 8000"

REM Wait for backend to start
timeout /t 3 /nobreak >nul

echo Starting Frontend on http://localhost:3000
start "Frontend Server" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo ==========================================
echo   Services Running!
echo ==========================================
echo.
echo   Backend API:  http://localhost:8000
echo   Frontend UI:  http://localhost:3000
echo   API Docs:     http://localhost:8000/docs
echo.
echo Close the terminal windows to stop the services.
echo.
pause
