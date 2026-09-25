#!/usr/bin/env bash
# The read-only Observatory (127.0.0.1:8095) and its public HTTPS quick tunnel.
#   scripts/observatory.sh start | stop | status | url
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-status}" in
  start)
    pgrep -f "^.venv/bin/python -m observatory.server" >/dev/null || \
      (setsid nohup .venv/bin/python -m observatory.server > /tmp/observatory.log 2>&1 &)
    pgrep -f "cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8095" >/dev/null || \
      (setsid nohup "$HOME/bin/cloudflared" tunnel --no-autoupdate --url http://127.0.0.1:8095 > /tmp/cloudflared.log 2>&1 &)
    ;;
  stop)
    for p in $(pgrep -f "^.venv/bin/python -m observatory.server") $(pgrep -f "cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8095"); do
      kill "$p"
    done
    ;;
  status) pgrep -af "observatory.server|cloudflared tunnel" || echo "not running" ;;
  url) grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /tmp/cloudflared.log | head -1 ;;
esac
