#!/usr/bin/env bash
# Downloads logs/ and results_*.csv from a Runpod benchmark machine via scp.
#
# Runpod's ssh.runpod.io proxy only supports interactive PTY sessions, not
# the exec/subsystem channels scp/sftp need — so this uses the pod's direct
# TCP SSH connection instead (shown on the pod's Connect panel), e.g.:
#   root@69.30.85.236 -p 22134
#
# Uses legacy scp protocol (-O) because Runpod's minimal container images
# don't have the sftp-server subsystem enabled, and modern OpenSSH scp
# defaults to SFTP under the hood.
set -u

KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR="/llm-mp3-perf-test"

read -rp "Runpod direct SSH connection (e.g. root@69.30.85.236 -p 22134): " CONN

if [ -z "$CONN" ]; then
    echo "ERROR: no connection string given."
    exit 1
fi

# shellcheck disable=SC2206
CONN_PARTS=($CONN)
MACHINE="${CONN_PARTS[0]}"
PORT=22
for ((i = 1; i < ${#CONN_PARTS[@]}; i++)); do
    if [ "${CONN_PARTS[$i]}" = "-p" ]; then
        PORT="${CONN_PARTS[$((i + 1))]}"
    fi
done

HOST="${MACHINE#*@}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/downloaded/${HOST}_${PORT}"

mkdir -p "$LOCAL_DIR/logs"

SCP_OPTS=(-O -P "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new)

echo "Downloading from $MACHINE:$PORT:$REMOTE_DIR to $LOCAL_DIR ..."

echo ""
echo "--- Fetching results_*.csv ---"
if scp "${SCP_OPTS[@]}" "$MACHINE:$REMOTE_DIR/results_*.csv" "$LOCAL_DIR/"; then
    echo "OK: results_*.csv fetched"
else
    echo "WARNING: no results_*.csv found (or scp failed) — see error above"
fi

echo ""
echo "--- Fetching logs/*.log ---"
if scp "${SCP_OPTS[@]}" "$MACHINE:$REMOTE_DIR/logs/*.log" "$LOCAL_DIR/logs/"; then
    echo "OK: logs/*.log fetched"
else
    echo "WARNING: no logs/*.log found (or scp failed) — see error above"
fi

echo ""
echo "=== Done. Downloaded files: ==="
find "$LOCAL_DIR" -type f
