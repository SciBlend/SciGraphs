#!/usr/bin/env bash
# Open Blender with a JupyterLab server serving these notebooks.
#
#   ./notebooks/tools/launch.sh                       # port 10462, opens a browser
#   ./notebooks/tools/launch.sh 8899                  # another port
#   ./notebooks/tools/launch.sh 8899 --no-browser
#   ./notebooks/tools/launch.sh 10462 --opengl        # for POINT/DISK renders
#   ./notebooks/tools/launch.sh --keep                # do not close the previous instance
#
# By default it closes the Blender and the server left behind by the previous
# launch, because the extension only supports one kernel at a time (see below).
#
# The kernel lives inside the Blender process, so a cell that touches `bpy`
# changes the scene you are looking at. Closing Blender stops the server.
#
# Only 127.0.0.1 is bound. To reach it from another machine, tunnel:
#
#   ssh -N -L 10462:127.0.0.1:10462 user@this-host
#
# Binding to 0.0.0.0 would expose arbitrary Python execution on your machine to
# the whole network — the URL token is the only thing in the way. Don't, unless
# the network is one you control.

set -euo pipefail

PORT=10462
OPEN_BROWSER=""
CLEANUP="yes"
BACKEND=()
for arg in "$@"; do
  case "$arg" in
    --no-browser) OPEN_BROWSER="--no-browser" ;;
    # Vulkan (Blender's default on Linux) silently draws POINT/DISK nodes at
    # 1 px, because gpu.state.point_size_set is a no-op there for the add-on's
    # shaders. Pass --opengl if you plan to render with those styles.
    --opengl)     BACKEND=(--gpu-backend opengl) ;;
    --keep)       CLEANUP="no" ;;
    -*)           echo "unknown option: $arg"; exit 2 ;;
    *)            PORT="$arg" ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NOTEBOOKS="$(dirname "$HERE")"
BLENDER="${BLENDER:-blender}"
BLEND="${SCIGRAPHS_BLEND:-}"

if [[ "$OPEN_BROWSER" == "--no-browser" ]]; then
  OPERATOR="start_server_headless"
else
  OPERATOR="start_server_or_open_browser"
fi

# --- release the previous instance -------------------------------------------
#
# The extension only supports one kernel at a time: it stores the connection
# file at a fixed path, .../jupyter_blender/connection_cache/jupyter-blender-kernel.json,
# shared by every Blender in this installation. A second Blender overwrites it
# and the first one's server ends up pointing at a kernel that is no longer its
# own — the notebooks stop responding without saying why. Launching on another
# port does NOT avoid it.
#
# So rather than refusing to start, we close the previous one. Only processes
# this script started are touched (`blender ... --python /tmp/tmp.*.py`) plus
# the extension's JupyterLab servers; a Blender you opened by hand does not
# match the pattern and is left alone.
#
# --keep skips the cleanup.

kill_pids() {   # kill_pids <pid...>  — TERM, then KILL if still alive after 5 s
  local pids=("$@") alive=()
  [[ ${#pids[@]} -eq 0 ]] && return 0
  kill "${pids[@]}" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    alive=()
    for pid in "${pids[@]}"; do kill -0 "$pid" 2>/dev/null && alive+=("$pid"); done
    [[ ${#alive[@]} -eq 0 ]] && return 0
    sleep 1
  done
  kill -9 "${alive[@]}" 2>/dev/null || true
}

if [[ "$CLEANUP" == "yes" ]]; then
  mapfile -t PREVIOUS < <(pgrep -f 'blender.*--python /tmp/tmp\..*\.py' 2>/dev/null || true)
  mapfile -t SERVERS < <(pgrep -f '_jupyterlab_launcher\.py' 2>/dev/null || true)

  if [[ ${#PREVIOUS[@]} -gt 0 || ${#SERVERS[@]} -gt 0 ]]; then
    echo "Closing the previous instance:"
    for pid in "${PREVIOUS[@]}"; do echo "    blender    pid $pid"; done
    for pid in "${SERVERS[@]}";  do echo "    jupyterlab pid $pid"; done
    kill_pids "${SERVERS[@]}"
    kill_pids "${PREVIOUS[@]}"
  fi

  # The dead kernel's connection file: if it stays, the new server's
  # provisioner tries to attach to a kernel that no longer exists.
  # `|| true` because under `set -e` a glob with no matches would make the loop
  # body fail and abort the script before launching anything.
  for f in "$HOME"/.config/blender/*/extensions/.user/*/jupyter_blender/connection_cache/*.json; do
    if [[ -e "$f" ]]; then
      rm -f "$f" && echo "    connection file  $(basename "$f")"
    fi
  done || true
fi

# Wait until the port is really free: TIME_WAIT can hold it for a few seconds
# after the server closes, and binding over it fails.
for _ in 1 2 3 4 5 6 7 8 9 10; do
  ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN || break
  sleep 1
done

if ss -ltn "sport = :$PORT" 2>/dev/null | grep -q LISTEN; then
  echo "Port $PORT is still held by something this script did not start."
  ss -ltnp 2>/dev/null | grep ":$PORT " | sed 's/^/    /'
  echo "Pick another:  $0 $((PORT + 1))"
  exit 1
fi

BOOTSTRAP="$(mktemp --suffix=.py)"
cat > "$BOOTSTRAP" <<PY
import bpy

ADDON = "bl_ext.user_default.jupyter_blender"
prefs = bpy.context.preferences.addons[ADDON].preferences
prefs.host = "127.0.0.1"
prefs.port = $PORT
try:
    prefs.notebook_dir = r"$NOTEBOOKS"
except AttributeError:
    pass   # older builds of the extension serve from the Blender cwd

def start():
    bpy.ops.jupyter_blender.$OPERATOR()
    print("=" * 64)
    print("JupyterLab at http://127.0.0.1:$PORT/lab")
    print("Notebooks in  $NOTEBOOKS")
    print("If the browser does not open on its own, copy the tokenized URL from")
    print("the N sidebar -> Jupyter -> Copy URL.")
    print("=" * 64)
    return None

# The server is launched once the interface exists; doing it while the script
# is loading leaves the window half built.
bpy.app.timers.register(start, first_interval=1.5)
PY

echo "Blender  : $BLENDER"
echo "Port     : $PORT"
echo "Notebooks: $NOTEBOOKS"
echo

# The kernel inherits Blender's working directory — not the notebook's, which
# is what a normal Jupyter would do — so we start from `notebooks/`.
cd "$NOTEBOOKS"

# env -u LD_LIBRARY_PATH: the system TBB, which the OpenFOAM profile puts
# ahead, makes Blender abort on start-up.
if [[ -n "$BLEND" ]]; then
  exec env -u LD_LIBRARY_PATH "$BLENDER" "${BACKEND[@]}" "$BLEND" --python "$BOOTSTRAP"
else
  exec env -u LD_LIBRARY_PATH "$BLENDER" "${BACKEND[@]}" --python "$BOOTSTRAP"
fi
