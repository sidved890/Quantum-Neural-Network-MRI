#!/usr/bin/env bash

# Directory containing the files
dir="/Users/ayaansyed/Downloads/Ayaan-2025-07-13"

# Check that the directory exists
if [[ ! -d "$dir" ]]; then
  echo "Directory '$dir' does not exist."
  exit 1
fi

# Loop over each item in the directory
for filepath in "$dir"/*; do
  # Only process regular files
  if [[ -f "$filepath" ]]; then
    filename=$(basename "$filepath")
    newname="Ayaan---${filename}"
    # Rename the file by prefixing 'Ayaan---'
    mv -- "$filepath" "$dir/$newname"
    echo "Renamed '$filename' to '$newname'"
  fi
done

echo "All files have been prefixed with 'Ayaan---'."
