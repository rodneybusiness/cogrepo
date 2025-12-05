"""
CogRepo Gunicorn Configuration

Production-ready WSGI server configuration with:
- Worker management
- Connection handling
- Logging
- Security settings

Usage:
    gunicorn -c gunicorn.conf.py app:app

    Or with SocketIO:
    gunicorn -c gunicorn.conf.py --worker-class eventlet "app:app"
"""

import os
import multiprocessing

# =============================================================================
# Server Socket
# =============================================================================

# Bind to this address
bind = os.getenv('GUNICORN_BIND', '0.0.0.0:5001')

# Backlog - number of pending connections
backlog = 2048

# =============================================================================
# Workers
# =============================================================================

# Number of worker processes
# Recommendation: (2 x CPU cores) + 1
workers = int(os.getenv('GUNICORN_WORKERS', (multiprocessing.cpu_count() * 2) + 1))

# Worker class
# Use 'eventlet' or 'gevent' for WebSocket support
worker_class = os.getenv('GUNICORN_WORKER_CLASS', 'sync')

# Number of threads per worker (for sync workers)
threads = int(os.getenv('GUNICORN_THREADS', 4))

# Maximum concurrent connections per worker
worker_connections = 1000

# Maximum requests a worker will process before restarting
max_requests = int(os.getenv('GUNICORN_MAX_REQUESTS', 1000))

# Add jitter to max_requests to prevent all workers restarting at once
max_requests_jitter = int(os.getenv('GUNICORN_MAX_REQUESTS_JITTER', 100))

# =============================================================================
# Timeouts
# =============================================================================

# Worker timeout (seconds) - how long before killing a hung worker
timeout = int(os.getenv('GUNICORN_TIMEOUT', 120))

# Graceful timeout for workers to finish current requests
graceful_timeout = int(os.getenv('GUNICORN_GRACEFUL_TIMEOUT', 30))

# Keep-alive connections timeout
keepalive = int(os.getenv('GUNICORN_KEEPALIVE', 5))

# =============================================================================
# Process Naming
# =============================================================================

# Process name prefix
proc_name = 'cogrepo'

# =============================================================================
# Logging
# =============================================================================

# Access log file
accesslog = os.getenv('GUNICORN_ACCESS_LOG', '-')  # '-' = stdout

# Error log file
errorlog = os.getenv('GUNICORN_ERROR_LOG', '-')  # '-' = stderr

# Log level
loglevel = os.getenv('GUNICORN_LOG_LEVEL', 'info')

# Access log format
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Capture output from app
capture_output = True

# Enable access log
disable_redirect_access_to_syslog = True

# =============================================================================
# Security
# =============================================================================

# Limit request line length (URL + protocol)
limit_request_line = 4094

# Limit request fields (headers)
limit_request_fields = 100

# Limit header field size
limit_request_field_size = 8190

# =============================================================================
# Server Mechanics
# =============================================================================

# Daemonize the Gunicorn process (set True for production)
daemon = False

# PID file location
pidfile = os.getenv('GUNICORN_PID_FILE', None)

# User to run workers as (requires root)
# user = 'www-data'
# group = 'www-data'

# Temporary directory for worker heartbeat
worker_tmp_dir = '/dev/shm'

# =============================================================================
# SSL (optional - usually handled by reverse proxy)
# =============================================================================

# SSL certificate file
# certfile = '/path/to/cert.pem'

# SSL key file
# keyfile = '/path/to/key.pem'

# =============================================================================
# Hooks
# =============================================================================

def on_starting(server):
    """Called just before the master process is initialized."""
    print(f"[CogRepo] Starting Gunicorn server with {workers} workers")


def on_reload(server):
    """Called to recycle workers during a reload."""
    print("[CogRepo] Reloading workers...")


def worker_int(worker):
    """Called when a worker receives SIGINT or SIGQUIT."""
    print(f"[CogRepo] Worker {worker.pid} interrupted")


def worker_abort(worker):
    """Called when a worker receives SIGABRT."""
    print(f"[CogRepo] Worker {worker.pid} aborted")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    print(f"[CogRepo] Worker {worker.pid} spawned")


def post_worker_init(worker):
    """Called just after a worker has initialized."""
    pass


def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    print(f"[CogRepo] Worker {worker.pid} exited")


def nworkers_changed(server, new_value, old_value):
    """Called when number of workers changes."""
    print(f"[CogRepo] Workers changed from {old_value} to {new_value}")


def on_exit(server):
    """Called just before exiting Gunicorn."""
    print("[CogRepo] Gunicorn shutting down")
