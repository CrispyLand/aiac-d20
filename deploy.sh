#!/usr/bin/env bash
# Deploy both apps to the Hetzner VPS.
# Run from the repository root: ./deploy.sh
# Prerequisites (one-time, on the VPS):
#   echo 'GROQ_API_KEY=gsk_...' | ssh russell@2.28.124.111 'install -m 600 /dev/stdin /opt/aiac/.env'

set -euo pipefail

VPS="russell@2.28.124.111"
ROOT="$(cd "$(dirname "$0")" && pwd)"

# ── 1. Build ──────────────────────────────────────────────────────────────────
echo "==> Building agent..."
cd "$ROOT"
mvn -q package -DskipTests

echo "==> Building MCP server..."
cd "$ROOT/mcp-server"
mvn -q package -DskipTests
cd "$ROOT"

AGENT_JAR="$ROOT/target/agent-chat-0.0.1-SNAPSHOT.jar"
MCP_JAR="$ROOT/mcp-server/target/mcp-server-0.0.1-SNAPSHOT.jar"

# ── 2. Provision directories (idempotent) ─────────────────────────────────────
echo "==> Creating directories on VPS..."
ssh "$VPS" "mkdir -p /opt/aiac/mcp/data /opt/aiac/agent/data /opt/aiac/agent/profiles /opt/aiac/notes"

# ── 3. Install nginx (idempotent) ─────────────────────────────────────────────
echo "==> Ensuring nginx is installed..."
ssh "$VPS" "which nginx > /dev/null 2>&1 || sudo apt-get install -y nginx"

# ── 4. Upload jars ────────────────────────────────────────────────────────────
echo "==> Uploading agent jar ($(du -h "$AGENT_JAR" | cut -f1))..."
scp "$AGENT_JAR" "$VPS:/opt/aiac/agent/agent.jar"

echo "==> Uploading MCP server jar ($(du -h "$MCP_JAR" | cut -f1))..."
scp "$MCP_JAR" "$VPS:/opt/aiac/mcp/mcp-server.jar"

# ── 5. Upload Google credentials ──────────────────────────────────────────────
echo "==> Uploading credentials.json..."
scp "$ROOT/credentials.json" "$VPS:/opt/aiac/credentials.json"

echo "==> Syncing Google tokens (preserving permissions)..."
# rsync -a preserves the drwx------ that FileDataStoreFactory expects.
rsync -a "$ROOT/.google-tokens/" "$VPS:/opt/aiac/.google-tokens/"

# ── 6. Upload profiles ────────────────────────────────────────────────────────
if [ -d "$ROOT/profiles" ] && [ "$(ls -A "$ROOT/profiles")" ]; then
    echo "==> Syncing profiles..."
    rsync -a "$ROOT/profiles/" "$VPS:/opt/aiac/agent/profiles/"
fi

# ── 7. Install systemd units ──────────────────────────────────────────────────
echo "==> Installing systemd units..."
scp "$ROOT/deploy/aiac-mcp.service"   "$VPS:/tmp/aiac-mcp.service"
scp "$ROOT/deploy/aiac-agent.service" "$VPS:/tmp/aiac-agent.service"
ssh "$VPS" "
    sudo mv /tmp/aiac-mcp.service   /etc/systemd/system/aiac-mcp.service
    sudo mv /tmp/aiac-agent.service /etc/systemd/system/aiac-agent.service
    sudo systemctl daemon-reload
    sudo systemctl enable aiac-mcp aiac-agent
"

# ── 8. Install nginx config ───────────────────────────────────────────────────
echo "==> Installing nginx config..."
scp "$ROOT/deploy/nginx-aiac.conf" "$VPS:/tmp/nginx-aiac.conf"
ssh "$VPS" "
    sudo mv /tmp/nginx-aiac.conf /etc/nginx/sites-available/aiac
    sudo ln -sf /etc/nginx/sites-available/aiac /etc/nginx/sites-enabled/aiac
    sudo rm -f /etc/nginx/sites-enabled/default
    sudo nginx -t
    sudo systemctl reload nginx
"

# ── 9. Restart services (MCP first — agent's MCP client does not reconnect) ───
echo "==> Restarting MCP server..."
ssh "$VPS" "sudo systemctl restart aiac-mcp"

echo "==> Waiting for MCP server to open its port..."
ssh "$VPS" "
    for i in \$(seq 1 20); do
        ss -tlnp | grep -q ':8081' && echo 'Port 8081 is open.' && break
        echo \"  attempt \$i/20...\"
        sleep 2
    done
"

echo "==> Restarting agent..."
ssh "$VPS" "sudo systemctl restart aiac-agent"

# ── 10. Status ────────────────────────────────────────────────────────────────
echo ""
echo "==> Done. Services:"
ssh "$VPS" "sudo systemctl status aiac-mcp aiac-agent --no-pager -l | grep -E 'Loaded|Active|Main PID'"
echo ""
echo "Logs:  ssh $VPS 'journalctl -u aiac-mcp -u aiac-agent -f'"
echo "Agent: http://2.28.124.111"
