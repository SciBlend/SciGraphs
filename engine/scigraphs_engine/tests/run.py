# Every check for the public API, one process per file: they end in os._exit,
# and one that hangs an import or loses the GPU device must not pass that on.
# Each declares a minimum check count, since a file that died before reaching
# the code under test reports no failures either. SKIP is not a pass and is not
# counted.
#
#     python engine/scigraphs_engine/tests/run.py          # -v for full output

import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SKIP_CODE = 77

# test_readme.py is not here: the README no longer carries worked examples for
# it to run. The file is left in place for whenever it does again.
FILES = ("test_api.py", "test_parity.py", "test_degraded.py", "test_render.py")

# Duplicated on purpose: catches a file whose own minimum was lowered to pass.
EXPECTED_MINIMUM = {
    "test_api.py": 90,
    "test_parity.py": 60,
    "test_degraded.py": 80,
    "test_render.py": 19,
}

_PASSED = re.compile(r"^(\S+): (\d+) checks passed$", re.M)
_FAILED = re.compile(r"^(\S+): (\d+) of (\d+) FAILED$", re.M)
_TOOFEW = re.compile(r"^(\S+): ONLY (\d+) CHECKS RAN", re.M)
_SKIPPED = re.compile(r"^(\S+): SKIPPED -- (.*)$", re.M)


def run_one(name, verbose):
    path = os.path.join(HERE, name)
    proc = subprocess.run([sys.executable, path], capture_output=True,
                          text=True, cwd=os.environ.get("TMPDIR", "/tmp"))
    out = proc.stdout + proc.stderr
    if verbose:
        print(out)

    skipped = _SKIPPED.search(out)
    if proc.returncode == SKIP_CODE and skipped:
        print(f"  SKIP {name:<20} {skipped.group(2)}")
        return None, 0

    ran = failed = 0
    # The last summary line wins: test_degraded prints one per configuration.
    matches = _PASSED.findall(out)
    if matches:
        ran = int(matches[-1][1])
    else:
        matches = _FAILED.findall(out)
        if matches:
            failed, ran = int(matches[-1][1]), int(matches[-1][2])
        else:
            matches = _TOOFEW.findall(out)
            ran = int(matches[-1][1]) if matches else 0

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
    if "--only" in argv:
        files = [argv[argv.index("--only") + 1]]

    print(f"scigraphs_engine public API -- {len(files)} files")
    print(f"  python: {sys.executable}")
    print(f"  cwd:    {os.environ.get('TMPDIR', '/tmp')}  "
          f"(outside the repository, on purpose)")
    ok, total, skips = True, 0, []
    for name in files:
        good, ran = run_one(name, verbose)
        if good is None:
            skips.append(name)
            continue
        ok &= good
        total += ran

    print()
    tail = f" ({len(skips)} file(s) SKIPPED: {', '.join(skips)})" if skips else ""
    if ok:
        print(f"ALL {total} CHECKS PASSED across "
              f"{len(files) - len(skips)} files{tail}")
        return 0
    print(f"FAILURES; {total} checks ran across "
          f"{len(files) - len(skips)} files{tail}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
