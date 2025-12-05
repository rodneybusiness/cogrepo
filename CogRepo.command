#!/bin/bash
# CogRepo - Launch Web UI
# Double-click this file or search "CogRepo" in Spotlight/Alfred/Raycast

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Load environment
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Check dependencies silently
python3 -c "import flask, flask_socketio, flask_cors" 2>/dev/null || {
    echo "Installing dependencies..."
    pip3 install -q flask flask-socketio flask-cors python-dotenv
}

# Open browser after short delay (server needs time to start)
(sleep 2 && open "http://localhost:5000") &

# Start server
cd cogrepo-ui
echo "CogRepo starting at http://localhost:5000"
echo "Press Ctrl+C to stop"
python3 app.py
