# The house check(name, ok, detail) pattern and the counter behind it. A copy
# of backends/wgpu/tests/harness.py rather than an import: that package's
# __init__ imports wgpu, which these tests must not require. Every file
# declares a floor, so a file that checked nothing can't look green, and skip()
# exits 77, which run.py prints as SKIP rather than OK.

import os
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

SKIP_CODE = 77


class Report:
    def __init__(self, name, minimum=1):
        self.name = name
        self.minimum = int(minimum)
        self.total = 0
        self.failed = []
        self.notes = []

    def check(self, name, ok, detail=""):
        self.total += 1
        print(f"  {'OK  ' if ok else 'FAIL'} {name}"
              f"{'  ' + str(detail) if detail else ''}")
        if not ok:
            self.failed.append(name)
        return bool(ok)

    def equal(self, name, got, want):
        """Exact compare; a bare False is not debuggable, so report shapes."""
        import numpy as np
        a, b = np.asarray(got), np.asarray(want)
        if a.shape != b.shape:
            return self.check(name, False, f"shape {a.shape} != {b.shape}")
        same = bool(np.array_equal(a, b))
        detail = f"{a.shape} identical" if same else \
            f"{int((a != b).sum())} of {a.size} entries differ"
        return self.check(name, same, detail)

    def close(self, name, got, want, tol=0.0):
        import numpy as np
        a, b = np.asarray(got, np.float64), np.asarray(want, np.float64)
        if a.shape != b.shape:
            return self.check(name, False, f"shape {a.shape} != {b.shape}")
        if a.size == 0:
            return self.check(name, True, "both empty")
        worst = float(np.nanmax(np.abs(a - b)))
        return self.check(name, worst <= tol, f"max |diff| {worst:.3g}")

    def note(self, text):
        """Something measured but not asserted. Never a verdict."""
        self.notes.append(text)
        print(f"  ..   {text}")

    def section(self, text):
        print(f"\n{text}")

    def raises(self, name, exc_type, fn, *args, **kwargs):
        """Any other exception is itself a failure: a catch-everything except
        would pass on an ImportError from a typo."""
        try:
            fn(*args, **kwargs)
        except exc_type as exc:
            return self.check(name, True, f"raised {type(exc).__name__}")
        except Exception as exc:    # noqa: BLE001 - that IS the failure
            return self.check(name, False,
                              f"raised {type(exc).__name__}, wanted "
                              f"{exc_type.__name__}: {exc}")
        return self.check(name, False, f"did not raise {exc_type.__name__}")

    @property
    def ok(self):
        return not self.failed and self.total >= self.minimum

    def summary(self):
        if self.total < self.minimum:
            print(f"\n{self.name}: ONLY {self.total} CHECKS RAN, expected at "
                  f"least {self.minimum} -- the file did not reach the code "
                  f"under test")
            return 1
        if self.failed:
            print(f"\n{self.name}: {len(self.failed)} of {self.total} FAILED")
            for name in self.failed:
                print(f"  - {name}")
            return 1
        print(f"\n{self.name}: {self.total} checks passed")
        return 0


def finish(report):
    """Print the summary and leave. Never returns."""
    code = report.summary()
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(code)


def skip(name, reason):
    """Leave without pretending anything was checked. Never returns."""
    print(f"\n{name}: SKIPPED -- {reason}")
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(SKIP_CODE)
