#!/bin/bash

# GNN-DRL Frontend - Start Development Server

set -e

echo "🚀 Starting GNN-DRL Frontend Development Server"
echo "=============================================="
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Dependencies not installed. Running setup..."
    ./setup.sh
fi

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env from .env.example..."
    cp .env.example .env
fi

# Check backend
BACKEND_URL=${VITE_API_BASE_URL:-http://localhost:8000}
echo "🔌 Checking backend at $BACKEND_URL..."

if curl -s -f -o /dev/null "$BACKEND_URL/health"; then
    echo "✓ Backend is running"
else
    echo "⚠️  Warning: Backend is not reachable"
    echo "   Start backend with: cd ../backend && python main.py api --port 8000"
fi

echo ""
echo "🌐 Starting Vite development server..."
echo "   Frontend will be available at: http://localhost:3000"
echo "   Backend API proxy: $BACKEND_URL"
echo ""
echo "Press Ctrl+C to stop"
echo ""

npm run dev
