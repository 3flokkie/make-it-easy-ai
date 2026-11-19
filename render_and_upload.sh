#!/bin/bash
set -e
export PATH=$PATH:/usr/local/bin:/opt/homebrew/bin

COMP_NAME=${1:-"HorrorPart4Movie"}
OUT_FILE=${2:-"/Users/sashasyrota/PycharmProjects/WelcomeScreen/video-engine/out/horror_part4_nosiren.mp4"}

echo "=== 1. Waiting for render to finish if running, or verifying file ==="
if [ ! -f "$OUT_FILE" ]; then
    echo "Rendering $COMP_NAME to $OUT_FILE..."
    cd /Users/sashasyrota/PycharmProjects/WelcomeScreen/video-engine
    npx remotion render "$COMP_NAME" "$OUT_FILE" --timeout=120000 --concurrency=2
fi

echo "=== 2. Auto-uploading directly to Google Drive ==="
UPLOAD_OUT=$(gdrive mutate upload "$OUT_FILE")
echo "$UPLOAD_OUT"

FILE_ID=$(echo "$UPLOAD_OUT" | grep -o 'ID: [a-zA-Z0-9_-]*' | cut -d' ' -f2)

if [ -n "$FILE_ID" ]; then
    echo "=== 3. Enabling public access ==="
    gdrive mutate share "$FILE_ID" --role reader --type anyone --notify=false
    echo "=== 4. Direct Drive URL ==="
    gdrive readonly url "$FILE_ID"
fi
