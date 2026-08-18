"""Execute each notebook source inside a real Blender and report what happened.

    python3 notebooks/tools/verify_notebooks.py [04 05] [--offline]

One Blender per notebook, so a crash cannot take the rest down, and errors go to
a JSON report, not to stdout, which Blender floods after a failure. It drops
LD_LIBRARY_PATH (a system TBB aborts start-up with an undefined symbol) and ends
in `os._exit` (teardown segfaults with this add-on and would return 134).
"""

import json
import os
import pathlib
import subprocess
import sys

TOOLS_DIR = pathlib.Path(__file__).resolve().parent
NOTEBOOKS_DIR = TOOLS_DIR.parent
SRC_DIR = NOTEBOOKS_DIR / "_src"
REPORT_DIR = NOTEBOOKS_DIR / "out" / "verify"

# Networks reached, listed not parsed so a new download is a deliberate edit.
NEEDS_NETWORK = {"13_morphology", "16_proximity", "17_metapaths_15min",
                 "18_case_study", "19_repro_pipelines",
                 # 20 also pulls a model if none is installed.
                 "20_pipeline_assistant",
                 "11_terrain", "12_imagery",
                 "06_osmnx_download", "07_osmnx_geometry", "08_osmnx_routing",
                 "09_osmnx_accessibility", "10_osmnx_centrality_export"}

RUNNER = r'''
import json, os, runpy, sys, tempfile, traceback, time
sys.stdout.reconfigure(line_buffering=True)
src, report_path, notebooks_dir = sys.argv[sys.argv.index("--") + 1: sys.argv.index("--") + 4]

# Reproduce how a notebook really sees the world.
#
# The kernel lives inside Blender and was created before any notebook was
# opened, so its cwd is wherever Blender was launched from, NOT the notebook's
# directory. Running from an unrelated directory makes a bootstrap that quietly
# depends on the cwd fail here instead of in the user's browser. (It did once:
# chdir'ing into notebooks/ hid exactly that bug.)
os.chdir(tempfile.mkdtemp(prefix="scigraphs-nb-cwd-"))

# What the notebooks *can* rely on is the directory JupyterLab is serving,
# which launch.sh writes into the extension's preferences. Set it here too.
try:
    import bpy
    bpy.context.preferences.addons[
        "bl_ext.user_default.jupyter_blender"].preferences.notebook_dir = notebooks_dir
except Exception as exc:
    print("could not set notebook_dir (%s); the cwd fallback is on its own" % exc)
result = {"source": src, "ok": False, "error": None, "seconds": 0.0}
started = time.time()
try:
    runpy.run_path(src, run_name="__verify__")
    result["ok"] = True
except BaseException:
    result["error"] = traceback.format_exc()
result["seconds"] = round(time.time() - started, 1)
os.makedirs(os.path.dirname(report_path), exist_ok=True)
with open(report_path, "w", encoding="utf-8") as handle:
    json.dump(result, handle, indent=2)
print("VERIFY", "ok" if result["ok"] else "FAILED", src)
sys.stdout.flush(); sys.stderr.flush()
os._exit(0 if result["ok"] else 1)
'''


def blender():
    return os.environ.get("BLENDER", "blender")


def run_one(source, timeout=1800):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{source.stem}.json"
    runner = REPORT_DIR / "_runner.py"
    runner.write_text(RUNNER, encoding="utf-8")

    env = dict(os.environ)
    env.pop("LD_LIBRARY_PATH", None)

    # --gpu-backend opengl is not optional for anything that renders: on the
    # Vulkan backend (the Linux default) `gpu.state.point_size_set` is a no-op
    # for custom shaders and POINT/DISK nodes rasterize at 1 pixel.
    command = [blender(), "-b", "--gpu-backend", "opengl",
               "--python", str(runner), "--",
               str(source), str(report_path), str(NOTEBOOKS_DIR)]
    log_path = REPORT_DIR / f"{source.stem}.log"
    # Grandchildren hold the inherited stdout, so `capture_output=True` hangs.
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            subprocess.run(command, env=env, timeout=timeout,
                           stdout=log, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        return {"source": source.name, "ok": False,
                "error": f"timed out after {timeout}s", "seconds": timeout,
                "log": str(log_path)}
    tail = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]

    if report_path.exists():
        result = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        result = {"source": source.name, "ok": False,
                  "error": "Blender produced no report, it probably crashed",
                  "seconds": 0.0}
    result["source"] = source.name
    result["tail"] = tail
    return result


def main(argv):
    offline = "--offline" in argv
    wanted = tuple(a for a in argv if not a.startswith("-"))

    sources = sorted(SRC_DIR.glob("*.py"))
    if wanted:
        sources = [p for p in sources if p.stem.startswith(wanted)]
    if offline:
        sources = [p for p in sources if p.stem not in NEEDS_NETWORK]
    if not sources:
        print("nothing to verify")
        return 1

    results = []
    for source in sources:
        print(f"--- {source.name}", flush=True)
        result = run_one(source)
        results.append(result)
        mark = "ok  " if result["ok"] else "FAIL"
        print(f"    [{mark}] {result['seconds']}s", flush=True)
        if not result["ok"]:
            error = (result.get("error") or "").strip().splitlines()
            for line in error[-6:]:
                print("      " + line, flush=True)

    summary = REPORT_DIR / "summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")

    failed = [r for r in results if not r["ok"]]
    print()
    print(f"{len(results) - len(failed)}/{len(results)} notebooks ran clean")
    print("report:", summary)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
