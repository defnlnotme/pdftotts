#!/bin/bash
# Exit immediately if a command exits with a non-zero status.
set -e

# Function to output error messages.
error_exit() {
  echo "Error: $1"
  exit 1
}

# Check that the ~/dev/owl repo is present and is a git repository.
if [ ! -d "$HOME/dev/owl/.git" ]; then
  error_exit "Git repository not found in ~/dev/owl"
fi

# Check that the ~/dev/pdftotts repo is present and is a git repository.
if [ ! -d "$HOME/dev/pdftotts/.git" ]; then
  error_exit "Git repository not found in ~/dev/pdftotts"
fi

# Change directory to ~/dev/owl
cd "$HOME/dev/owl" || error_exit "Failed to change directory to ~/dev/owl"

# Activate the virtual environment.
if [ -f ".venv/bin/activate" ]; then
  # Shellcheck source=venv/bin/activate
  source ".venv/bin/activate"
else
  error_exit "Virtual environment not found in ~/dev/owl/venv"
fi

# Call the pdftotts.py script located in ~/dev/pdftotts and pass all arguments.
python "$HOME/dev/pdftotts/pdftotts.py" "$@"
