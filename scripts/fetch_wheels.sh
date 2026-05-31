#!/bin/bash
set -euo pipefail

# Change to project root directory (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p wheels

PYVER=3.13
PIP="python3 -m pip"

# Download wheels for a constraints file. Accepts one or more platform tags;
# all of them are passed to a SINGLE `pip download` invocation so the resolver
# can satisfy each package from whichever tag matches. This is required because
# pip does NOT expand an explicit `--platform manylinux_2_28` to also accept
# `manylinux_2_17` wheels, so packages tagged at different manylinux levels
# (e.g. scigraphs-utils @ 2_28 vs scipy @ 2_17) must be resolved together.
_download() {
	local constraints="$1"
	local dest_dir="$2"
	shift 2
	local platform_args=()
	local tag
	for tag in "$@"; do
		platform_args+=(--platform "${tag}")
	done
	${PIP} download -r "${constraints}" --dest "${dest_dir}" --only-binary=:all: \
		--python-version=${PYVER} "${platform_args[@]}" || true
}

echo "Downloading Linux x64 wheels (manylinux_2_28 + manylinux_2_17)..."
_download constraints/linux-x64.txt ./wheels \
	manylinux_2_28_x86_64 manylinux_2_17_x86_64 manylinux2014_x86_64

echo "Downloading Windows x64 wheels..."
_download constraints/windows-x64.txt ./wheels win_amd64

echo "Downloading macOS ARM64 wheels..."
_download constraints/macos-arm64.txt ./wheels \
	macosx_14_0_arm64 macosx_12_0_arm64 macosx_11_0_arm64

echo "Cleaning up unwanted wheels..."
find ./wheels -type f -name 'numpy-*.whl' -print -delete || true

TOTAL_WHEELS=$(find ./wheels -type f -name '*.whl' | wc -l)
echo ""
echo "✓ Total wheels: $TOTAL_WHEELS"
echo "Wheel download complete!"
