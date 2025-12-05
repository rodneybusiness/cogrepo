# =============================================================================
# CogRepo Docker Image
# =============================================================================
#
# Multi-stage build for a lean production image
#
# Build:
#   docker build -t cogrepo:latest .
#
# Run:
#   docker run -p 5001:5001 -v $(pwd)/data:/app/data cogrepo:latest
#
# With environment:
#   docker run -p 5001:5001 \
#     -e ANTHROPIC_API_KEY=sk-ant-... \
#     -v $(pwd)/data:/app/data \
#     cogrepo:latest
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
WORKDIR /build

# Create requirements.txt for pip install
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi

# Install core dependencies directly
RUN pip install --no-cache-dir \
    flask>=2.3.0 \
    flask-cors>=4.0.0 \
    flask-socketio>=5.3.0 \
    python-dotenv>=1.0.0 \
    pyyaml>=6.0 \
    gunicorn>=21.0.0 \
    eventlet>=0.33.0 \
    psutil>=5.9.0 \
    pydantic>=2.0.0 \
    pydantic-settings>=2.0.0

# Install optional dependencies (may fail on some platforms)
RUN pip install --no-cache-dir numpy>=1.24.0 || true


# -----------------------------------------------------------------------------
# Stage 2: Production Image
# -----------------------------------------------------------------------------
FROM python:3.11-slim as production

# Labels
LABEL maintainer="CogRepo Team"
LABEL version="2.0.0"
LABEL description="CogRepo - Knowledge Repository for LLM Conversations"

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FLASK_ENV=production \
    PORT=5001

# Create non-root user
RUN groupadd -r cogrepo && useradd -r -g cogrepo cogrepo

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create app directory structure
WORKDIR /app

# Copy application code
COPY cogrepo-ui/ ./cogrepo-ui/
COPY core/ ./core/
COPY database/ ./database/
COPY enrichment/ ./enrichment/
COPY search/ ./search/
COPY context/ ./context/
COPY parsers/ ./parsers/
COPY models/ ./models/
COPY state_manager.py ./
COPY cogrepo_import.py ./

# Copy config (templates only, not secrets)
COPY config/*.yaml ./config/

# Create data directory
RUN mkdir -p ./data && chown -R cogrepo:cogrepo /app

# Copy startup script
COPY --chmod=755 docker-entrypoint.sh /docker-entrypoint.sh

# Switch to non-root user
USER cogrepo

# Expose port
EXPOSE 5001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5001/health || exit 1

# Default command
ENTRYPOINT ["/docker-entrypoint.sh"]
CMD ["gunicorn", "-c", "cogrepo-ui/gunicorn.conf.py", "--worker-class", "eventlet", "cogrepo-ui.app:app"]
