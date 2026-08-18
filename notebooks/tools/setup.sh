#!/usr/bin/env bash
# Install everything needed to run these notebooks against a live Blender.
#
#   ./notebooks/tools/setup.sh
#
# Two things get installed, both inside Blender. Nothing touches the system
# Python:
#
#   1. The `jupyter_blender` extension, which embeds an IPython kernel in the
#      Blender process and launches a JupyterLab server that attaches to it.
#   2. That extension's Python dependencies (JupyterLab + ipykernel, ~100 MB),
#      into Blender's own extension site-packages. The extension carries no
#      wheels of its own, so step 2 is not optional. Checked against 0.1.11,
#      the newest release: its blender_manifest.toml has no `wheels` array,
#      and upstream still documents the same manual dependency install.
#
# The same extension is also published as a Blender repository at
# https://jan-hendrik-mueller.de/blender-extensions/index.json, addable under
# Preferences > Get Extensions > Repositories. That route tracks the latest
# version; this script pins one so a notebook run stays reproducible. Either
# way, step 2 still has to happen.
#
# Re-running is safe: the extension install overwrites, and pip skips what is
# already there.

set -euo pipefail

BLENDER="${BLENDER:-blender}"
VERSION="${JUPYTER_BLENDER_VERSION:-0.1.11}"
URL="https://github.com/kolibril13/jupyter-blender/releases/download/v${VERSION}/jupyter-blender-v${VERSION}.zip"
ZIP="$(mktemp -d)/jupyter-blender-v${VERSION}.zip"

# The OpenFOAM setup in this machine's profile puts the system TBB ahead of
# Blender's bundled one, and Blender then dies on start-up with an
# undefined-symbol error. Dropping the variable is the fix.
run_blender() { env -u LD_LIBRARY_PATH "$BLENDER" "$@"; }

echo "==> Blender"
run_blender --version | head -2

echo
echo "==> Downloading jupyter-blender ${VERSION}"
curl -fsSL -o "$ZIP" "$URL"
ls -lh "$ZIP" | awk '{print "    " $5, $9}'

echo
echo "==> Installing the extension"
run_blender --command extension install-file -r user_default -e "$ZIP" 2>&1 \
  | grep -E "STATUS|Error|error" || true

echo
echo "==> Installing JupyterLab and ipykernel into Blender's Python"
echo "    (~100 MB on first run; nothing to do afterwards)"
run_blender -b --python-expr "
import bpy, sys, os
result = bpy.ops.jupyter_blender.install_python_modules()
print('INSTALL', result)
sys.stdout.flush(); os._exit(0 if 'FINISHED' in result else 1)
" 2>&1 | grep -E "INSTALL|Error|error" || true

echo
echo "==> Checking the notebook dependencies inside Blender"
run_blender -b --python-expr "
import importlib, sys, os
missing = []
for name in ('city2graph', 'geopandas', 'osmnx', 'networkx', 'libpysal',
             'momepy', 'shapely', 'sklearn', 'jupyterlab', 'ipykernel'):
    try:
        mod = importlib.import_module(name)
        print('  ok      %-14s %s' % (name, getattr(mod, '__version__', '?')))
    except Exception:
        missing.append(name)
        print('  MISSING %s' % name)
sys.stdout.flush(); os._exit(1 if missing else 0)
" 2>&1 | grep -E "^  (ok|MISSING)"

echo
echo "Done. Start the notebooks with:"
echo
echo "    ./notebooks/tools/launch.sh"
echo
