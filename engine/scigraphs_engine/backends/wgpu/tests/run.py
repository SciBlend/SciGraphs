# Every check in this backend, in one command.
#
#     /tmp/eng-venv/bin/python engine/scigraphs_engine/backends/wgpu/tests/run.py
#     ...                                                             run.py --bench
#
# Each file runs in its own process, so a lost wgpu device is not inherited,
# and declares a check floor: green cannot tell "passed" from "did not run".

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FILES = ("test_device.py", "test_shaders.py", "test_upload.py",
         "test_pipelines.py", "test_render.py")
BENCH = ("bench.py",)

# Restated here so run.py catches a floor lowered in the file itself.
EXPECTED_MINIMUM = {
    "test_device.py": 18,
    "test_shaders.py": 45,
    "test_upload.py": 50,
    "test_pipelines.py": 25,
    "test_render.py": 35,
    "bench.py": 6,
}

_PASSED = re.compile(r"^(\S+): (\d+) checks passed$", re.M)
_FAILED = re.compile(r"^(\S+): (\d+) of (\d+) FAILED$", re.M)
_TOOFEW = re.compile(r"^(\S+): ONLY (\d+) CHECKS RAN", re.M)


def run_one(name, verbose):
    path = os.path.join(HERE, name)
    proc = subprocess.run([sys.executable, path], capture_output=True,
                          text=True)
    out = proc.stdout + proc.stderr
    if verbose:
        print(out)

    ran = failed = 0
    m = _PASSED.search(out)
    if m:
        ran = int(m.group(2))
    else:
        m = _FAILED.search(out)
        if m:
            failed, ran = int(m.group(2)), int(m.group(3))
        else:
            m = _TOOFEW.search(out)
            ran = int(m.group(2)) if m else 0

    floor = EXPECTED_MINIMUM.get(name, 1)
    status = "OK"
    detail = f"{ran} checks"
    if proc.returncode != 0:
        status = "FAIL"
        detail = (f"{failed} of {ran} failed" if failed
                  else f"exit {proc.returncode}, {ran} checks")
    elif ran < floor:
        status = "FAIL"
        detail = (f"only {ran} checks ran, run.py expects at least {floor} -- "
                  f"the file did not reach the code under test")
    print(f"  {status:<4} {name:<20} {detail}")
    if status != "OK" and not verbose:
        print("      " + "\n      ".join(out.strip().splitlines()[-25:]))
    return status == "OK", ran


def main(argv):
    verbose = "-v" in argv or "--verbose" in argv
    files = list(FILES)
    if "--bench" in argv:
        files += list(BENCH)
    if "--only" in argv:
        files = [argv[argv.index("--only") + 1]]

    print(f"scigraphs_engine.backends.wgpu -- {len(files)} files")
    print(f"  python: {sys.executable}")
    ok, total = True, 0
    for name in files:
        good, ran = run_one(name, verbose)
        ok &= good
        total += ran

    print()
    if ok:
        print(f"ALL {total} CHECKS PASSED across {len(files)} files")
        return 0
    print(f"FAILURES; {total} checks ran across {len(files)} files")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
