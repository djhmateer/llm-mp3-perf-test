#!/usr/bin/env bash
# Downloads logs/ and results_*.csv from a Runpod benchmark machine via scp.
#
# Runpod's ssh.runpod.io proxy only supports interactive PTY sessions, not
# the exec/subsystem channels scp/sftp need — so this uses the pod's direct
# TCP SSH connection instead (shown on the pod's Connect panel). Accepts
# either form:
#   root@69.30.85.236 -p 22134
#   ssh root@69.30.85.236 -p 22134 -i ~/.ssh/id_ed25519
#
# Uses legacy scp protocol (-O) because Runpod's minimal container images
# don't have the sftp-server subsystem enabled, and modern OpenSSH scp
# defaults to SFTP under the hood.
set -u

KEY="$HOME/.ssh/id_ed25519"
REMOTE_DIR_CANDIDATES=("/llm-mp3-perf-test" "/root/llm-mp3-perf-test")

read -rp "Runpod direct SSH connection (e.g. ssh root@69.30.85.236 -p 22134 -i ~/.ssh/id_ed25519): " CONN

if [ -z "$CONN" ]; then
    echo "ERROR: no connection string given."
    exit 1
fi

# shellcheck disable=SC2206
CONN_PARTS=($CONN)

MACHINE=""
PORT=22
for ((i = 0; i < ${#CONN_PARTS[@]}; i++)); do
    part="${CONN_PARTS[$i]}"
    case "$part" in
        ssh) ;;
        -p) PORT="${CONN_PARTS[$((i + 1))]}" ;;
        -i) KEY="${CONN_PARTS[$((i + 1))]/#\~/$HOME}" ;;
        -*) ;;
        *)
            if [ -z "$MACHINE" ] && [ "${CONN_PARTS[$((i - 1))]}" != "-p" ] && [ "${CONN_PARTS[$((i - 1))]}" != "-i" ]; then
                MACHINE="$part"
            fi
            ;;
    esac
done

if [ -z "$MACHINE" ]; then
    echo "ERROR: could not find a user@host in the connection string."
    exit 1
fi

HOST="${MACHINE#*@}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/downloaded/${HOST}_${PORT}"

mkdir -p "$LOCAL_DIR/logs"

SCP_OPTS=(-O -P "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new)
SSH_OPTS=(-p "$PORT" -i "$KEY" -o StrictHostKeyChecking=accept-new)

REMOTE_DIR=""
for candidate in "${REMOTE_DIR_CANDIDATES[@]}"; do
    if ssh "${SSH_OPTS[@]}" "$MACHINE" "[ -d '$candidate' ]" 2>/dev/null; then
        REMOTE_DIR="$candidate"
        break
    fi
done

if [ -z "$REMOTE_DIR" ]; then
    echo "ERROR: couldn't find the repo at any of: ${REMOTE_DIR_CANDIDATES[*]}"
    exit 1
fi

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
