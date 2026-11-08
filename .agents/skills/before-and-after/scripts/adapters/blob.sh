#!/bin/bash
# Custom blob storage adapter - For self-hosted or custom endpoints
#
# Usage: ./blob.sh <file>
# Output: URL to uploaded file (stdout)
#
# Environment:
#   BLOB_UPLOAD_URL    Required. URL endpoint for uploading images
#
# Notes:
#   - Expects endpoint to accept multipart form upload with "file" field
#   - Expects response to be either:
#     - Plain text URL
#     - JSON with "url" field

set -e

FILE="$1"

if [[ -z "$FILE" ]]; then
    echo "Usage: $0 <file>" >&2
    exit 1
fi

if [[ ! -f "$FILE" ]]; then
    echo "Error: File not found: $FILE" >&2
    exit 1
fi

if [[ -z "$BLOB_UPLOAD_URL" ]]; then
    echo "Error: BLOB_UPLOAD_URL environment variable not set" >&2
    echo "" >&2
    echo "Set it with:" >&2
    echo "  export BLOB_UPLOAD_URL='https://your-blob-service.com/upload'" >&2
    exit 1
fi

# Upload file
RESPONSE=$(curl -s -X POST -F "file=@$FILE" "$BLOB_UPLOAD_URL")

# Plain text responses already contain the URL. Parse JSON responses as data so
# escaped URL characters are decoded before validation.
URL="$RESPONSE"
if [[ ! "$RESPONSE" =~ ^https?:// ]]; then
    URL=$(printf '%s' "$RESPONSE" | node -e '
const fs = require("node:fs");
try {
    const response = JSON.parse(fs.readFileSync(0, "utf8"));
    if (response === null || Array.isArray(response) || typeof response !== "object" || typeof response.url !== "string") {
        process.exit(1);
    }
    process.stdout.write(response.url);
} catch {
    process.exit(1);
}
') || URL=""
fi

# Validate we got a URL back
if [[ ! "$URL" =~ ^https?:// ]]; then
    echo "Error: Upload failed. Response: $RESPONSE" >&2
    exit 1
fi

echo "$URL"
