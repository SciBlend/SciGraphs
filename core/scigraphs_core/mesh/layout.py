# Compatibility facade: the implementation lives in SciGraphs.core.mesh.layouts,
# split by algorithm family. Kept so `from SciGraphs.core.mesh import layout` and
# `from SciGraphs.core import layout` keep working.

from .layouts import *
