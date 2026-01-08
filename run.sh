#!/bin/bash

# =============================================================================
# Smart B-Roll Inserter - Linux/Mac Startup Script
# =============================================================================

echo "=========================================="
echo "  Smart B-Roll Inserter - Starting..."
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# =============================================================================
# Check Python
# =============================================================================
echo -e "${YELLOW}Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo -e "${RED}Error: Python is not installed. Please install Python 3.8+${NC}"
    exit 1
fi

echo -e "${GREEN}Using: $($PYTHON_CMD --version)${NC}"

# =============================================================================
# Check Node.js
# =============================================================================
echo -e "${YELLOW}Checking Node.js installation...${NC}"
if ! command -v node &> /dev/null; then
    echo -e "${RED}Error: Node.js is not installed. Please install Node.js 16+${NC}"
    exit 1
fi
echo -e "${GREEN}Using: Node $(node --version)${NC}"

# =============================================================================
# Setup Python Virtual Environment
# =============================================================================
echo -e "${YELLOW}Setting up Python virtual environment...${NC}"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# =============================================================================
# Install Backend Dependencies
# =============================================================================
echo -e "${YELLOW}Installing backend dependencies...${NC}"
cd backend
pip install -r requirements.txt --quiet
cd ..

# =============================================================================
# Install Frontend Dependencies
# =============================================================================
echo -e "${YELLOW}Installing frontend dependencies...${NC}"
cd frontend
if [ ! -d "node_modules" ]; then
    npm install
fi
cd ..

# =============================================================================
# Create .env file if not exists
# =============================================================================
if [ ! -f "backend/.env" ]; then
    echo -e "${YELLOW}Creating .env file...${NC}"
    cat > backend/.env << EOF
# API Provider: "openai", "gemini", "openrouter", or "offline"
API_PROVIDER=offline

# Google Gemini API Key (get from https://makersuite.google.com/app/apikey)
GEMINI_API_KEY=

# OpenAI API Key (optional)
OPENAI_API_KEY=

# OpenRouter API Key (optional - for access to multiple models)
OPENROUTER_API_KEY=
EOF
    echo -e "${GREEN}.env file created. Edit backend/.env to add your API keys.${NC}"
fi

# =============================================================================
# Start Services
# =============================================================================
echo ""
echo -e "${GREEN}=========================================="
echo "  Starting Services..."
echo "==========================================${NC}"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start Backend
echo -e "${GREEN}Starting Backend on http://localhost:8000${NC}"
cd backend
$PYTHON_CMD -m uvicorn app:app --reload --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 3

# Start Frontend
echo -e "${GREEN}Starting Frontend on http://localhost:3001${NC}"
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo -e "${GREEN}=========================================="
echo "  Services Running!"
echo "==========================================${NC}"
echo ""
echo -e "  Backend API:  ${GREEN}http://localhost:8000${NC}"
echo -e "  Frontend UI:  ${GREEN}http://localhost:3000${NC}"
echo -e "  API Docs:     ${GREEN}http://localhost:8000/docs${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for processes
wait
