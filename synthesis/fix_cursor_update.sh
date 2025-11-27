#!/bin/bash
# Fix Cursor AppImage update issues
# Run with: sudo bash fix_cursor_update.sh

set -e

CURSOR_BIN="/home/chung/.local/bin/cursor"
CURSOR_DIR="/home/chung/.local/bin"

echo "🔧 Fixing Cursor update issues..."

# Remove partial downloads and old update files
echo "📦 Cleaning up partial downloads..."
rm -f "$CURSOR_DIR/cursor.part"
rm -f "$CURSOR_DIR/cursor.zs-old"
rm -f "$CURSOR_DIR/cursor.zsync"

# Fix ownership if needed
if [ -f "$CURSOR_BIN" ]; then
    CURRENT_OWNER=$(stat -c "%U:%G" "$CURSOR_BIN")
    if [ "$CURRENT_OWNER" != "chung:chung" ]; then
        echo "🔐 Fixing ownership of $CURSOR_BIN..."
        chown chung:chung "$CURSOR_BIN"
        echo "✅ Ownership changed from $CURRENT_OWNER to chung:chung"
    else
        echo "✅ Ownership is already correct (chung:chung)"
    fi
    
    # Ensure executable permissions
    chmod +x "$CURSOR_BIN"
    echo "✅ Executable permissions set"
else
    echo "⚠️  Cursor binary not found at $CURSOR_BIN"
fi

echo ""
echo "✨ Cleanup complete! Cursor should now update properly."
echo "💡 Restart Cursor to trigger a new update attempt."

