#!/bin/bash
# LAVENDER — Service Installation
# Run this once to register Lavender as a systemd service.
# After this, Lavender starts automatically on boot.
#
# Usage: sudo ./scripts/install_service.sh
# ─────────────────────────────────────────────────────────────────────────────

set -e

LAVENDER_ROOT="/opt/lavender"
SERVICE_NAME="lavender"
SERVICE_FILE="$LAVENDER_ROOT/scripts/lavender.service"
SYSTEMD_DIR="/etc/systemd/system"

# ── VERIFY ROOT ───────────────────────────────────────────────────────────────
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run as root (sudo ./scripts/install_service.sh)"
    exit 1
fi

echo ""
echo "  Lavender — Service Installation"
echo "  ─────────────────────────────────────────────────────────────────────"
echo ""

# ── CREATE SERVICE USER ──────────────────────────────────────────────────────
echo "[1/6] Creating service user 'lavender'..."

if id "lavender" &>/dev/null; then
    echo "    ✓ User 'lavender' already exists."
else
    useradd \
        --system \
        --no-create-home \
        --shell /usr/sbin/nologin \
        --groups audio,video,input,render \
        lavender
    echo "    ✓ User 'lavender' created."
fi

# Add current user to lavender group for file access
REAL_USER="${SUDO_USER:-$USER}"
if [ -n "$REAL_USER" ] && [ "$REAL_USER" != "root" ]; then
    usermod -aG lavender "$REAL_USER" 2>/dev/null || true
    echo "    ✓ Added '$REAL_USER' to lavender group."
fi

# ── SET PERMISSIONS ───────────────────────────────────────────────────────────
echo ""
echo "[2/6] Setting directory permissions..."

# Lavender user needs to write logs and memory
chown -R lavender:lavender "$LAVENDER_ROOT/logs"  2>/dev/null || mkdir -p "$LAVENDER_ROOT/logs" && chown lavender:lavender "$LAVENDER_ROOT/logs"
chown -R lavender:lavender "$LAVENDER_ROOT/memory" 2>/dev/null || mkdir -p "$LAVENDER_ROOT/memory" && chown lavender:lavender "$LAVENDER_ROOT/memory"

# Read access to everything else
chmod -R a+rX "$LAVENDER_ROOT"
chmod 600 "$LAVENDER_ROOT/config/.env" 2>/dev/null || true

echo "    ✓ Permissions set."

# ── AUDIO PERMISSIONS ─────────────────────────────────────────────────────────
echo ""
echo "[3/6] Configuring audio access..."

# Allow lavender user to access audio devices
if [ -f /etc/pulse/client.conf ]; then
    # PulseAudio — add lavender to the pulse-access group
    groupadd -f pulse-access
    usermod -aG pulse-access lavender
fi

# ALSA permissions
if [ -f /etc/group ]; then
    usermod -aG audio lavender
fi

echo "    ✓ Audio access configured."

# ── INSTALL SERVICE FILE ──────────────────────────────────────────────────────
echo ""
echo "[4/6] Installing systemd service..."

if [ ! -f "$SERVICE_FILE" ]; then
    echo "    ERROR: Service file not found at $SERVICE_FILE"
    echo "    Run setup.sh first."
    exit 1
fi

cp "$SERVICE_FILE" "$SYSTEMD_DIR/${SERVICE_NAME}.service"
chmod 644 "$SYSTEMD_DIR/${SERVICE_NAME}.service"

echo "    ✓ Service file installed to $SYSTEMD_DIR/${SERVICE_NAME}.service"

# ── RELOAD AND ENABLE ────────────────────────────────────────────────────────
echo ""
echo "[5/6] Enabling service..."

systemctl daemon-reload
systemctl enable "${SERVICE_NAME}.service"

echo "    ✓ Service enabled (will start on boot)."

# ── LOG ROTATION ──────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Setting up log rotation..."

cat > /etc/logrotate.d/lavender << 'LOGROTATE'
$(pwd)/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 640 lavender lavender
    postrotate
        systemctl reload lavender 2>/dev/null || true
    endscript
}
LOGROTATE

echo "    ✓ Log rotation configured (14-day rolling)."

# ── DONE ─────────────────────────────────────────────────────────────────────
echo ""
echo "  ─────────────────────────────────────────────────────────────────────"
echo "  Service installed successfully."
echo ""
echo "  Commands:"
echo "    sudo systemctl start lavender      — start now"
echo "    sudo systemctl stop lavender       — stop"
echo "    sudo systemctl restart lavender    — restart"
echo "    sudo systemctl status lavender     — check status"
echo "    journalctl -u lavender -f          — live logs"
echo ""
echo "  Lavender will now start automatically on boot."
echo "  To start it now:"
echo "    sudo systemctl start lavender"
echo "  ─────────────────────────────────────────────────────────────────────"
echo ""
