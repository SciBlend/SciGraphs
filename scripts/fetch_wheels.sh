#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

mkdir -p wheels

PYVER=3.13
PIP="python3 -m pip"

# All platform tags go to ONE pip download call. pip will not expand an explicit
# --platform manylinux_2_28 to also accept manylinux_2_17, so packages tagged at
# different levels have to be resolved together.
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

# The passes above prefer mysql-connector-python's compiled wheel (16-34 MB)
# over the pure-python one (~400 kB), and the C extension is not needed. Version
# stays pinned in the constraints file; it is only read back here.
MYSQL_REQ=$(grep -iE '^mysql-connector-python==' constraints/linux-x64.txt | head -n1)
echo "Downloading pure-python mysql-connector-python wheel (${MYSQL_REQ})..."
${PIP} download "${MYSQL_REQ}" --dest ./wheels --only-binary=:all: \
	--no-deps --implementation py --python-version=${PYVER} --abi none --platform any || true

echo "Cleaning up unwanted wheels..."
find ./wheels -type f -name 'numpy-*.whl' -print -delete || true
find ./wheels -type f -name 'requests-*.whl' -print -delete || true
find ./wheels -type f -name 'mysql_connector_python-*' ! -name '*py2.py3-none-any.whl' -print -delete || true

# setuptools walks the source directory, so a __pycache__ left by a plain import
# gets packaged: scigraphs_engine 0.1.0 shipped 26 .pyc built for CPython 3.12
# into an add-on running 3.13. Sweep first, then check rather than trust.
clean_pycache() {
	find "$1" -type d -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null || true
	# setuptools stages under build/lib and newer versions zip it wholesale, so
	# a stale copy carries old bytecode in: core/build/lib held 78 .pyc.
	rm -rf "$1/build"
	# egg-info holds a PKG-INFO from an older pyproject and README, and
	# setuptools reuses it, so a deleted comment can come straight back.
	rm -rf "$1"/*.egg-info
}

assert_no_pycache() {
	${PYTHON:-python3} - "$1" <<-'EOF'
	import sys, zipfile
	bad = [n for n in zipfile.ZipFile(sys.argv[1]).namelist()
	       if "__pycache__" in n or n.endswith(".pyc")]
	if bad:
	    sys.exit(f"{sys.argv[1]}: {len(bad)} bytecode entries packaged, "
	             f"first is {bad[0]}")
	EOF
}

# Built from engine/ rather than downloaded, so the wheel matches this working
# tree instead of the last release. Pure Python over numpy, so one py3-none-any
# wheel covers every platform above. The guard is for a partial checkout, where
# build_extension.sh drops the manifest line. Always rebuilt: a stale wheel
# imports fine and nobody notices.
if [ -d engine ]; then
	echo "Building scigraphs-engine wheel from engine/..."
	find ./wheels -type f -name 'scigraphs_engine-*.whl' -delete || true
	clean_pycache ./engine
	# --no-build-isolation keeps this offline-capable; setuptools is the only
	# build requirement and the running interpreter already has it.
	${PIP} wheel ./engine --no-deps --no-build-isolation --wheel-dir ./wheels
	assert_no_pycache "$(ls -1 ./wheels/scigraphs_engine-*.whl | head -1)"
fi

# The same for the analysis half. core/ is in every checkout that has the
# add-on; the guard is kept so a partial one fails the same way, not a new way.
if [ -d core ]; then
	echo "Building scigraphs-core wheel from core/..."
	find ./wheels -type f -name 'scigraphs_core-*.whl' -delete || true
	clean_pycache ./core
	${PIP} wheel ./core --no-deps --no-build-isolation --wheel-dir ./wheels
	assert_no_pycache "$(ls -1 ./wheels/scigraphs_core-*.whl | head -1)"
fi

TOTAL_WHEELS=$(find ./wheels -type f -name '*.whl' | wc -l)
echo ""
echo "✓ Total wheels: $TOTAL_WHEELS"
echo "Wheel download complete!"
