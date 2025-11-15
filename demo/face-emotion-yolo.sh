#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Construct absolute paths with the script directory fetched from the previous step
VENV_DIR="$SCRIPT_DIR/../face-emotion-yolo-venv"   # Path to the virtual environment
PYTHON_APP="$SCRIPT_DIR/../app-pt.py"  # Path to the Python script

# Change to parent directory (project root) so Flask can find everything
# Not necessary but safer if Flask wants to find files from the source directly
cd "$SCRIPT_DIR/.."

# ---------------- Activate virtual environment ----------------
if [ -f "$VENV_DIR/bin/activate" ]; then
    echo "Activating virtual environment..."
    source "$VENV_DIR/bin/activate"
else
    echo "Virtual environment not found at $VENV_DIR"
    exit 1
fi

# ---------------- Run Python app ----------------
echo "Starting Python application..."
python "$PYTHON_APP"

# ---------------- Optional: Deactivate ----------------
# deactivate
