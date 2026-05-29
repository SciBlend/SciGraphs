#!/bin/bash
set -euo pipefail

# Change to project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p wheels

PYVER=3.13
PIP="python3 -m pip"

_download() {
	local_constraints="$1"
	platform_tag="$2"
	dest_dir="$3"
	${PIP} download -r "${local_constraints}" --dest "${dest_dir}" --only-binary=:all: \
		--python-version=${PYVER} --platform="${platform_tag}" || true
}

echo "Downloading Linux x64 wheels (manylinux_2_28)..."
_download constraints/linux-x64.txt manylinux_2_28_x86_64 ./wheels

echo "Downloading Linux x64 wheels (manylinux_2_17)..."
_download constraints/linux-x64.txt manylinux_2_17_x86_64 ./wheels

echo "Downloading Windows x64 wheels..."
_download constraints/windows-x64.txt win_amd64 ./wheels

echo "Downloading macOS x64 wheels..."
_download constraints/macos-x64.txt macosx_13_0_x86_64 ./wheels

echo "Downloading macOS ARM64 wheels..."
_download constraints/macos-arm64.txt macosx_14_0_arm64 ./wheels

echo "Cleaning up unwanted wheels..."
find ./wheels -type f -name 'numpy-*.whl' -print -delete || true

TOTAL_WHEELS=$(find ./wheels -type f -name '*.whl' | wc -l)
echo ""
echo "✓ Total wheels: $TOTAL_WHEELS"
echo "Wheel download complete!"
