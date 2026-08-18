# Simple logging utility for SciGraphs

import os

# Set SCIGRAPHS_QUIET=1 to silence output, which is what the test suite does.
QUIET_MODE = os.environ.get('SCIGRAPHS_QUIET', '0') == '1'

def log(message):
    if not QUIET_MODE:
        print(message)

def set_quiet(quiet=True):
    global QUIET_MODE
    QUIET_MODE = quiet

