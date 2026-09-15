# Locate scigraphs_core: installed wheel first, then ``<repo>/core`` checkout.
# Separate module so scripts/tests can put the checkout on sys.path without
# importing the add-on root.

import importlib
import os
import sys


def locate():
    """Return scigraphs_core, installed or from the checkout."""
    try:
        return importlib.import_module("scigraphs_core")
    except ModuleNotFoundError:
        pass

    if os.environ.get("SCIGRAPHS_DEV_CHECKOUT") != "1":
        raise ModuleNotFoundError(
            "scigraphs_core is not installed. In an installed extension it "
            "ships as a wheel; rebuild with ./build_extension.sh. To run from "
            "a checkout instead, set SCIGRAPHS_DEV_CHECKOUT=1.")

    here = os.path.dirname(os.path.abspath(__file__))     # <add-on>
    repo_root = os.path.dirname(here)
    checkout = os.path.join(repo_root, "core")

    if os.path.isfile(os.path.join(checkout, "scigraphs_core", "__init__.py")):
        if checkout not in sys.path:
            sys.path.insert(0, checkout)
        return importlib.import_module("scigraphs_core")

    raise ModuleNotFoundError(
        "scigraphs_core is not installed and no checkout copy was found "
        f"(looked for {os.path.join(checkout, 'scigraphs_core')}).")
