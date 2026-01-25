#!/bin/bash
# Run expert analysis on a profile
# Usage: ./run_expert_analysis.sh profile.nsys-rep

set -e

PROFILE="${1:-profile.nsys-rep}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=================================================="
echo "Nsight Systems Expert Analysis"
echo "=================================================="
echo "Profile: $PROFILE"

# Check if profile exists
if [ ! -f "$PROFILE" ]; then
    echo "Error: Profile not found: $PROFILE"
    exit 1
fi

# Export to SQLite
SQLITE_FILE="${PROFILE%.nsys-rep}.sqlite"
echo ""
echo "Step 1: Exporting to SQLite..."
nsys export --type=sqlite -o "$SQLITE_FILE" "$PROFILE" --force

# Run custom expert analysis
echo ""
echo "Step 2: Running custom expert analysis..."
python3 "$SCRIPT_DIR/custom_expert_rules.py" "$SQLITE_FILE"

# Also run nsys stats for comparison
echo ""
echo "Step 3: Nsight Stats Summary..."
nsys stats -r gpukernsum "$PROFILE" 2>/dev/null | head -30

echo ""
echo "Analysis complete!"
echo "Files created:"
echo "  - $SQLITE_FILE (for custom queries)"
