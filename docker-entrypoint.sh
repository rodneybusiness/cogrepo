#!/bin/bash
# =============================================================================
# CogRepo Docker Entrypoint Script
# =============================================================================
#
# Handles:
# - Environment validation
# - Database initialization
# - Graceful shutdown
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=============================================="
echo "  CogRepo Container Starting..."
echo "=============================================="

# -----------------------------------------------------------------------------
# Environment Validation
# -----------------------------------------------------------------------------

echo -e "${GREEN}[1/4]${NC} Checking environment..."

# Check for API key (optional but recommended)
if [ -n "$ANTHROPIC_API_KEY" ]; then
    echo "  - Anthropic API key: configured"
else
    echo -e "  - Anthropic API key: ${YELLOW}not set${NC} (enrichment disabled)"
fi

# Check for data directory
if [ -d "/app/data" ]; then
    echo "  - Data directory: /app/data (exists)"
else
    echo "  - Data directory: creating /app/data"
    mkdir -p /app/data
fi

# Check for config
if [ -f "/app/config/enrichment_config.yaml" ]; then
    echo "  - Config file: found"
else
    echo "  - Config file: using defaults"
fi

# -----------------------------------------------------------------------------
# Database Initialization
# -----------------------------------------------------------------------------

echo -e "${GREEN}[2/4]${NC} Checking database..."

# Check if SQLite database exists
if [ -f "/app/data/cogrepo.db" ]; then
    echo "  - Database: found at /app/data/cogrepo.db"
else
    echo "  - Database: not found (will be created on first import)"
fi

# Check if JSONL data exists
if [ -f "/app/data/enriched_repository.jsonl" ]; then
    CONV_COUNT=$(wc -l < /app/data/enriched_repository.jsonl)
    echo "  - Data file: found with $CONV_COUNT conversations"
else
    echo "  - Data file: not found (upload data to get started)"
fi

# -----------------------------------------------------------------------------
# Port Configuration
# -----------------------------------------------------------------------------

echo -e "${GREEN}[3/4]${NC} Configuring network..."

PORT=${PORT:-5001}
echo "  - Listening on port: $PORT"

# Update Gunicorn bind if PORT is set
export GUNICORN_BIND="0.0.0.0:$PORT"

# -----------------------------------------------------------------------------
# Signal Handling
# -----------------------------------------------------------------------------

echo -e "${GREEN}[4/4]${NC} Setting up signal handlers..."

# Handle shutdown signals gracefully
trap 'echo "Received SIGTERM, shutting down gracefully..."; kill -TERM $PID; wait $PID' TERM INT

# -----------------------------------------------------------------------------
# Start Application
# -----------------------------------------------------------------------------

echo ""
echo "=============================================="
echo "  Starting CogRepo Server"
echo "=============================================="
echo ""
echo "  URL: http://localhost:$PORT"
echo "  Health: http://localhost:$PORT/health"
echo "  API v2: http://localhost:$PORT/api/v2/health"
echo ""
echo "=============================================="
echo ""

# Change to app directory
cd /app

# Execute the command
exec "$@"
