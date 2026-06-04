#!/bin/bash
set -e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXTENSION_ID="scigraphs"
BUILD_DIR="build_temp"

# Detect Blender config directory per-platform
if [ "$(uname)" = "Darwin" ]; then
    BLENDER_CONFIG_BASE="$HOME/Library/Application Support/Blender"
else
    BLENDER_CONFIG_BASE="${XDG_CONFIG_HOME:-$HOME/.config}/blender"
fi

# Allow overriding the target Blender version (default: auto-detect from manifest)
BLENDER_VERSION="${BLENDER_TARGET_VERSION:-}"

# ---------------------------------------------------------------------------
# Parse flags
# ---------------------------------------------------------------------------
SKIP_INSTALL=false
for arg in "$@"; do
    case "$arg" in
        --no-install) SKIP_INSTALL=true ;;
    esac
done

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
echo "=== SciGraphs Extension Build ==="
echo ""

echo "[1/5] Preparing build directory..."
rm -rf "$BUILD_DIR" dist
mkdir -p "$BUILD_DIR"

if [ ! -f "blender_manifest.toml" ]; then
    echo "  blender_manifest.toml not found"; exit 1
fi
cp blender_manifest.toml "$BUILD_DIR/"

if [ ! -d "SciGraphs" ]; then
    echo "  SciGraphs/ directory not found"; exit 1
fi
cp -r SciGraphs/* "$BUILD_DIR/"

if [ -d "wheels" ] && [ "$(ls -A wheels 2>/dev/null)" ]; then
    echo "  Copying wheels referenced in manifest..."
    mkdir -p "$BUILD_DIR/wheels"
    MISSING_WHEELS=false

    # Resolve the platform family of a wheel filename's platform tag (the
    # last '-'-delimited field). Used to reconcile manylinux tag-spelling
    # differences between environments (e.g. pip fetching a manylinux_2_28
    # wheel where the manifest pins the manylinux2014/2_17 spelling, or vice
    # versa). PyPI publishes both for some packages and the resolver may pick
    # either depending on platform-tag priority and pip version.
    _wheel_family() {
        case "$1" in
            *manylinux*|*musllinux*) echo "linux" ;;
            *win*) echo "win" ;;
            *macos*) echo "macos" ;;
            *any*) echo "any" ;;
            *) echo "" ;;
        esac
    }

    # Find an alternative wheel for the same distribution/version/python/abi
    # whose platform tag belongs to the same family as the requested one.
    _find_alt_wheel() {
        local wanted="$1"
        local prefix="${wanted%-*}"          # strip platform tag field
        local wanted_tag="${wanted##*-}"
        local family
        family="$(_wheel_family "$wanted_tag")"
        local cand base ct
        for cand in wheels/"$prefix"-*.whl; do
            [ -e "$cand" ] || continue
            base="$(basename "$cand")"
            ct="${base##*-}"
            if [ "$(_wheel_family "$ct")" = "$family" ]; then
                # Prefer manylinux over musllinux for linux requests.
                if [ "$family" = "linux" ]; then
                    case "$ct" in *manylinux*) echo "$base"; return 0 ;; esac
                else
                    echo "$base"; return 0
                fi
            fi
        done
        return 1
    }

    while IFS= read -r wheel; do
        if [ -f "wheels/$wheel" ]; then
            cp "wheels/$wheel" "$BUILD_DIR/wheels/"
            continue
        fi

        alt="$(_find_alt_wheel "$wheel" || true)"
        if [ -n "$alt" ] && [ -f "wheels/$alt" ]; then
            echo "  NOTE: reconciling '$wheel' -> '$alt'"
            cp "wheels/$alt" "$BUILD_DIR/wheels/"
            sed -i "s|./wheels/$wheel|./wheels/$alt|" "$BUILD_DIR/blender_manifest.toml"
        else
            echo "  WARNING: manifest references missing wheel: $wheel"
            MISSING_WHEELS=true
        fi
    done < <(grep -oP '\./wheels/\K[^"]+\.whl' blender_manifest.toml)

    if [ "$MISSING_WHEELS" = true ]; then
        echo "  Some manifest wheels are missing. Run scripts/fetch_wheels.sh."; exit 1
    fi
else
    echo "  No wheels found. Run scripts/fetch_wheels.sh first."
    mkdir -p "$BUILD_DIR/wheels"
fi

# ---------------------------------------------------------------------------
echo "[2/5] Resolving Blender binary..."
if [ -n "$BLENDER_BIN" ]; then
    if [[ "$BLENDER_BIN" != /* ]]; then
        BLENDER_CMD="$(cd "$(dirname "$BLENDER_BIN")" && pwd)/$(basename "$BLENDER_BIN")"
    else
        BLENDER_CMD="$BLENDER_BIN"
    fi
    if [ ! -f "$BLENDER_CMD" ]; then
        echo "  BLENDER_BIN not found: $BLENDER_CMD"
        if command -v blender &> /dev/null; then
            BLENDER_CMD="blender"
        else
            echo "  Blender not found"; exit 1
        fi
    fi
    echo "  Using: $BLENDER_CMD"
elif command -v blender &> /dev/null; then
    BLENDER_CMD="blender"
    echo "  Using: blender (from PATH)"
else
    echo "  Blender not found. Set BLENDER_BIN or add blender to PATH."
    exit 1
fi

# ---------------------------------------------------------------------------
echo "[3/5] Building extension..."
cd "$BUILD_DIR"
mkdir -p ../dist

"$BLENDER_CMD" --command extension build --source-dir . --output-dir ../dist --split-platforms

cd ..

echo ""
echo "  Built artifacts:"
ls -lh dist/

# ---------------------------------------------------------------------------
# Auto-install
# ---------------------------------------------------------------------------
if [ "$SKIP_INSTALL" = true ]; then
    echo ""
    echo "  Skipping install (--no-install)."
    echo "=== Done ==="
    exit 0
fi

echo ""
echo "[4/5] Detecting Blender version..."

# Read blender_version_min from manifest to find the right config folder
if [ -z "$BLENDER_VERSION" ]; then
    MANIFEST_VER=$(grep -oP 'blender_version_min\s*=\s*"\K[0-9]+\.[0-9]+' blender_manifest.toml 2>/dev/null || true)
    if [ -n "$MANIFEST_VER" ]; then
        BLENDER_VERSION="$MANIFEST_VER"
    fi
fi

if [ -z "$BLENDER_VERSION" ]; then
    echo "  Could not detect Blender version from manifest."
    echo "  Set BLENDER_TARGET_VERSION=X.Y to specify manually."
    echo "  Skipping auto-install."
    echo "=== Done (build only) ==="
    exit 0
fi

INSTALL_DIR="$BLENDER_CONFIG_BASE/$BLENDER_VERSION/extensions/user_default/$EXTENSION_ID"
echo "  Blender version: $BLENDER_VERSION"
echo "  Install target:  $INSTALL_DIR"

# Determine which zip to install based on current platform
PLATFORM="$(uname -s)-$(uname -m)"
case "$PLATFORM" in
    Linux-x86_64)  ZIP_SUFFIX="linux_x64" ;;
    Darwin-arm64)  ZIP_SUFFIX="macos_arm64" ;;
    Darwin-x86_64) echo "  macOS Intel (x86_64) is no longer supported (no native wheels for scigraphs-utils / pysurprise)."
                   echo "  SciGraphs requires Apple Silicon (arm64) on macOS."
                   exit 1 ;;
    *)             ZIP_SUFFIX="linux_x64"
                   echo "  Unknown platform '$PLATFORM', defaulting to linux_x64" ;;
esac

ZIP_FILE="dist/${EXTENSION_ID}-1.0.0-${ZIP_SUFFIX}.zip"
if [ ! -f "$ZIP_FILE" ]; then
    echo "  Expected zip not found: $ZIP_FILE"
    echo "  Skipping auto-install."
    echo "=== Done (build only) ==="
    exit 0
fi

echo ""
echo "[5/5] Installing to Blender $BLENDER_VERSION..."

# Clean previous installation completely
if [ -d "$INSTALL_DIR" ]; then
    echo "  Removing old installation..."
    rm -rf "$INSTALL_DIR"
fi

mkdir -p "$INSTALL_DIR"
unzip -q -o "$ZIP_FILE" -d "$INSTALL_DIR"

# Verify the installed manifest is correct
INSTALLED_VER=$(grep -oP 'blender_version_min\s*=\s*"\K[^"]+' "$INSTALL_DIR/blender_manifest.toml" 2>/dev/null || echo "UNKNOWN")
CP311_COUNT=$(grep -c "cp311" "$INSTALL_DIR/blender_manifest.toml" 2>/dev/null; true)

if [ "$CP311_COUNT" -gt 0 ]; then
    echo "ERROR: Installed manifest still contains cp311 wheels!"
    echo "  Something is wrong with the source blender_manifest.toml."
    exit 1
fi

echo "Installed successfully"
echo "     blender_version_min = \"$INSTALLED_VER\""
echo "     cp311 references: $CP311_COUNT"
echo ""
echo "=== Done! Open Blender $BLENDER_VERSION and enable SciGraphs. ==="
