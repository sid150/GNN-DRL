#!/bin/bash

# GNN-DRL Frontend - Production Build and Deployment

set -e

echo "🏗️  GNN-DRL Frontend - Production Build"
echo "======================================="
echo ""

# Install dependencies
echo "📦 Installing dependencies..."
npm ci --production=false

# Type check
echo "🔍 Running type check..."
npm run type-check

# Lint
echo "🧹 Running linter..."
npm run lint

# Build
echo "📦 Building for production..."
npm run build

echo ""
echo "✅ Build complete!"
echo ""
echo "Build output: dist/"
echo ""
echo "To preview the build:"
echo "  npm run preview"
echo ""
echo "To deploy:"
echo "  1. Upload dist/ directory to your web server"
echo "  2. Configure server to serve index.html for all routes (SPA)"
echo "  3. Update VITE_API_BASE_URL to production backend URL"
echo ""
