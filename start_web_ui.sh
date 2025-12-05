#!/bin/bash
# Start CogRepo Web UI with Upload Support

echo "======================================"
echo "  Starting CogRepo Web UI"
echo "======================================"
echo ""

# Navigate to cogrepo-ui directory
cd "$(dirname "$0")/cogrepo-ui"

# Check if API key is set
if [ -f "../.env" ]; then
    echo "✓ Found .env file"
    source ../.env
else
    echo "⚠  Warning: No .env file found"
    echo "  Create one with: echo 'ANTHROPIC_API_KEY=your-key' > .env"
fi

# Check Python dependencies
echo "Checking dependencies..."
python3 -c "import flask, flask_socketio, flask_cors" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing dependencies..."
    pip install -q flask flask-socketio flask-cors werkzeug --ignore-installed blinker
fi

echo ""
echo "Starting server..."
echo ""
echo "  📤 Upload UI: http://localhost:5000/upload.html"
echo "  🔍 Search UI: http://localhost:5000/index.html"
echo "  📊 API: http://localhost:5000/api/status"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start the server
python3 app.py
