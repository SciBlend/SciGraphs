# check() and the counter behind it. Every file declares a floor and run.py
# fails a file that ran fewer: passing over zero checks proves nothing.

import os
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


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

    def note(self, text):
        """A measured value. Printed, never part of the verdict."""
        self.notes.append(text)
        print(f"  ..   {text}")

    def section(self, text):
        print(f"\n{text}")

    def raises(self, name, exc_type, fn, *args, **kwargs):
        """Fails with whatever fn raised instead, so the detail is readable."""
        try:
            fn(*args, **kwargs)
        except exc_type as exc:
            return self.check(name, True, f"raised {type(exc).__name__}")
        except Exception as exc:    # noqa: BLE001 - that is the failure
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
    """Exits the process; never returns."""
    code = report.summary()
    try:
        sys.stdout.flush()
    except Exception:
        pass
    os._exit(code)
