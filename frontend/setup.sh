#!/bin/bash

# GNN-DRL Frontend - Development Setup Script

set -e

echo "🚀 GNN-DRL Frontend Setup"
echo "=========================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Node.js version
echo "📦 Checking prerequisites..."
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    echo "Please install Node.js >= 18.0.0 from https://nodejs.org/"
    exit 1
fi

NODE_VERSION=$(node -v | cut -d'v' -f2 | cut -d'.' -f1)
if [ "$NODE_VERSION" -lt 18 ]; then
    echo -e "${RED}❌ Node.js version must be >= 18.0.0${NC}"
    echo "Current version: $(node -v)"
    exit 1
fi

echo -e "${GREEN}✓ Node.js $(node -v)${NC}"
echo -e "${GREEN}✓ npm $(npm -v)${NC}"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dependencies installed${NC}"
echo ""

# Setup environment file
echo "⚙️  Setting up environment..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo -e "${GREEN}✓ Created .env file from .env.example${NC}"
    echo -e "${YELLOW}⚠️  Please review .env and update if needed${NC}"
else
    echo -e "${YELLOW}⚠️  .env file already exists, skipping${NC}"
fi
echo ""

# Check backend connection
echo "🔌 Checking backend connection..."
BACKEND_URL=${VITE_API_BASE_URL:-http://localhost:8000}

if curl -s -f -o /dev/null "$BACKEND_URL/health"; then
    echo -e "${GREEN}✓ Backend is running at $BACKEND_URL${NC}"
else
    echo -e "${YELLOW}⚠️  Backend is not reachable at $BACKEND_URL${NC}"
    echo "Make sure to start the backend before running the frontend"
fi
echo ""

# Summary
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Review and update .env if needed"
echo "  2. Ensure backend is running (cd ../backend && python main.py api)"
echo "  3. Start development server: npm run dev"
echo "  4. Open http://localhost:3000 in your browser"
echo ""
echo "Available commands:"
echo "  npm run dev       - Start development server"
echo "  npm run build     - Build for production"
echo "  npm run preview   - Preview production build"
echo "  npm run lint      - Run ESLint"
echo "  npm test          - Run tests"
echo ""
