#!/usr/bin/env bash
set -euo pipefail

#debugs : ssh tunnel skipped ( use export SSH_TARGET=$(hostname))

# Minimal helper: starts the streamer locally, optionally opens an SSH tunnel, and loads index.html.
# If SSH_TARGET is empty the tunnel step is skipped (handy when running on the remote host).

PYTHON_BIN=${PYTHON_BIN:-python}
SSH_TARGET=${SSH_TARGET:-}
PORT_HINT=${PORT:-}
PORT_START=${PORT_HINT:-8765}
# Local static HTTP server (for serving index.html over http:// instead of file://)
HTTP_PORT_HINT=${HTTP_PORT:-}
HTTP_PORT_START=${HTTP_PORT_HINT:-8770}
SSH_SOCKET_DIR=${SSH_SOCKET_DIR:-/tmp}
SSH_CTL=""
STRICT_FLAG=0
if [[ -n "$PORT_HINT" ]]; then
	STRICT_FLAG=1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HTML_FILE="${REPO_ROOT}/ws_stream/index.html"

SERVER_PID=""
SSH_PID=""
HTTP_PID=""

### AUTO-TUNNEL PATCH ###
if [[ -n "${SSH_CONNECTION:-}" && -z "${SSH_TARGET:-}" ]]; then
	if [[ -n "${SKYPILOT_CLUSTER_NAME:-}" ]]; then
		echo "[launch] Detected remote execution on SkyPilot — auto-enabling SSH tunnel"
		SSH_TARGET="${SKYPILOT_CLUSTER_NAME}"
	else
		echo "[launch] Remote execution detected but SSH_TARGET not set"
		echo "[launch] Set SSH_TARGET to your cluster hostname (e.g., SSH_TARGET=sky-91d3-anm)"
		echo "[launch] Skipping tunnel - browser may need manual WebSocket URL"
	fi
fi
### END PATCH ###

cleanup() {
	if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
		kill "$SERVER_PID" 2>/dev/null || true
		wait "$SERVER_PID" 2>/dev/null || true
	fi
	if [[ -n "$SSH_PID" ]] && kill -0 "$SSH_PID" 2>/dev/null; then
		kill "$SSH_PID" 2>/dev/null || true
	fi
	# Try graceful shutdown of SSH master connection if used
	if [[ -n "$SSH_CTL" && -S "$SSH_CTL" ]]; then
		ssh -S "$SSH_CTL" -O exit "${SSH_TARGET}" >/dev/null 2>&1 || true
		rm -f "$SSH_CTL" || true
	fi
	if [[ -n "$HTTP_PID" ]] && kill -0 "$HTTP_PID" 2>/dev/null; then
		kill "$HTTP_PID" 2>/dev/null || true
		wait "$HTTP_PID" 2>/dev/null || true
	fi
}

trap cleanup EXIT INT TERM

cd "$REPO_ROOT"

PORT=$(PORT_START="$PORT_START" PORT_HINT="$PORT_HINT" STRICT_FLAG="$STRICT_FLAG" "$PYTHON_BIN" - <<'PY'
import os
import socket
import sys

start_str = os.environ.get("PORT_START", "8765")
strict = bool(int(os.environ.get("STRICT_FLAG", "0")))
hint = os.environ.get("PORT_HINT", "")

try:
	start = int(start_str)
except ValueError:
	print(f"Invalid port hint: {start_str}", file=sys.stderr)
	sys.exit(1)

max_attempts = 50
for offset in range(max_attempts):
	port = start + offset
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		try:
			s.bind(('0.0.0.0', port))
		except OSError:
			if strict:
				print(f"Requested port {port} unavailable. Set PORT={hint} to a free port or unset it to auto-pick.", file=sys.stderr)
				sys.exit(1)
			continue
		else:
			print(port)
			break
else:
	print(f"Unable to find free port starting at {start}", file=sys.stderr)
	sys.exit(1)
PY
)

if [[ -z "$PORT" ]]; then
	echo "[launch] failed to determine a free port" >&2
	exit 1
fi

if [[ -n "$PORT_HINT" && "$PORT" != "$PORT_HINT" ]]; then
	echo "[launch] requested PORT=$PORT_HINT but using free port $PORT instead"
fi

echo "[launch] starting streamer on port $PORT"
"$PYTHON_BIN" -m ws_stream.server --use-tekken --codec webp --quality 65 --fps 30 --port "$PORT" &
SERVER_PID=$!

sleep 2

if [[ -n "$SSH_TARGET" ]]; then
	echo "[launch] opening ssh tunnel to $SSH_TARGET (local:${PORT} -> remote:localhost:${PORT})"
	SSH_CTL="${SSH_SOCKET_DIR%/}/owlwms_ssh_${PORT}.ctl"
	# Use ControlMaster so we can verify and close cleanly; fail fast if forwarding cannot be set up
	if ssh -M -S "$SSH_CTL" -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
		-f -N -L "${PORT}:localhost:${PORT}" "$SSH_TARGET"; then
		# Verify the control socket is active
		if ssh -S "$SSH_CTL" -O check "$SSH_TARGET" >/dev/null 2>&1; then
			echo "[launch] ssh tunnel established (control socket: $SSH_CTL)"
			# Best-effort to record a PID for visibility (optional)
			SSH_PID=$(pgrep -f "ssh .* -L ${PORT}:localhost:${PORT} .* ${SSH_TARGET}" | head -n1 || true)
		else
			echo "[launch] ssh tunnel may not be active (couldn't verify with control socket)" >&2
		fi
	else
		echo "[launch] failed to establish ssh tunnel to $SSH_TARGET" >&2
	fi
else
	echo "[launch] SSH_TARGET not set; skipping tunnel"
fi

# Pick an HTTP port for a tiny static file server
HTTP_PORT=$(HTTP_PORT_START="$HTTP_PORT_START" HTTP_PORT_HINT="$HTTP_PORT_HINT" STRICT_FLAG="$STRICT_FLAG" "$PYTHON_BIN" - <<'PY'
import os, socket, sys
start_str = os.environ.get("HTTP_PORT_START", "8770")
strict = bool(int(os.environ.get("STRICT_FLAG", "0")))
hint = os.environ.get("HTTP_PORT_HINT", "")
try:
	start = int(start_str)
except ValueError:
	print(f"Invalid HTTP port hint: {start_str}", file=sys.stderr)
	sys.exit(1)
max_attempts = 50
for offset in range(max_attempts):
	port = start + offset
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		try:
			s.bind(('127.0.0.1', port))
		except OSError:
			if strict:
				print(f"Requested HTTP port {port} unavailable. Set HTTP_PORT={hint} to a free port or unset it to auto-pick.", file=sys.stderr)
				sys.exit(1)
			continue
		else:
			print(port)
			break
else:
	print(f"Unable to find free HTTP port starting at {start}", file=sys.stderr)
	sys.exit(1)
PY
)

if [[ -z "$HTTP_PORT" ]]; then
	echo "[launch] failed to determine a free HTTP port" >&2
	exit 1
fi

echo "[launch] starting static file server on http://localhost:${HTTP_PORT} (serving repo root)"
"$PYTHON_BIN" -m http.server "$HTTP_PORT" --bind 127.0.0.1 --directory "$REPO_ROOT" >/dev/null 2>&1 &
HTTP_PID=$!
sleep 0.5

# Build an HTTP URL to the page with explicit WS port
BROWSER_URL="http://localhost:${HTTP_PORT}/ws_stream/index.html?port=${PORT}"
OPENED=0
if command -v xdg-open >/dev/null 2>&1; then
	echo "[launch] opening browser via xdg-open (${BROWSER_URL})"
	xdg-open "$BROWSER_URL" >/dev/null 2>&1 && OPENED=1 || true
fi
if [[ $OPENED -eq 0 ]]; then
	echo "[launch] trying python webbrowser (${BROWSER_URL})"
	"$PYTHON_BIN" -m webbrowser "$BROWSER_URL" >/dev/null 2>&1 && OPENED=1 || true
fi
if [[ $OPENED -eq 0 ]]; then
	echo "[launch] open this URL manually: $BROWSER_URL"
fi

echo "[launch] streamer active on port ${PORT}"
echo "[launch] WebSocket URL: ws://localhost:${PORT}"
echo "[launch] Page URL: ${BROWSER_URL}"
echo "[launch] Tip: If your browser shows a different port, ensure the page URL includes ?port=${PORT} or paste ws://localhost:${PORT} into the WS URL box."

wait "$SERVER_PID"