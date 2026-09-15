import sys
import importlib.util

# Layout algorithms that need scigraphs-utils, which is absent from the build
# published on extensions.blender.org: it bundles Graphviz under the EPL, and
# the EPL cannot be combined with Blender's GPL. The enum items stay in both
# builds so a .blend keeps its layout choice when moved between them.
GRAPHVIZ_ALGORITHMS = frozenset({
    'GRAPHVIZ_DOT', 'GRAPHVIZ_NEATO', 'GRAPHVIZ_FDP', 'GRAPHVIZ_SFDP',
    'GRAPHVIZ_TWOPI', 'GRAPHVIZ_CIRCO', 'GRAPHVIZ_OSAGE', 'GRAPHVIZ_PATCHWORK',
    'YIFAN_HU',
})

GRAPHVIZ_MISSING_MESSAGE = (
    "This layout needs scigraphs-utils, which the Blender Extensions build "
    "leaves out for licence reasons. The full build at "
    "github.com/SciBlend/SciGraphs/releases has it"
)

_graphviz_available = None


def graphviz_available():
    """Whether scigraphs-utils is installed. Resolved once."""
    global _graphviz_available
    if _graphviz_available is None:
        _graphviz_available = importlib.util.find_spec("scigraphs_utils") is not None
    return _graphviz_available


def check_dependencies():
    """
    Checks if all required dependencies are available.
    Returns a tuple (success, missing_packages).
    """
    required = ['numpy', 'pandas', 'networkx', 'scipy']
    missing = []
    
    for package in required:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing.append(package)
    
    return len(missing) == 0, missing

def get_dependency_status():
    """
    Returns a dictionary with the status of each dependency.
    """
    packages = ['numpy', 'pandas', 'networkx', 'scipy']
    status = {}
    
    for package in packages:
        spec = importlib.util.find_spec(package)
        if spec is not None:
            try:
                mod = importlib.import_module(package)
                version = getattr(mod, '__version__', 'unknown')
                status[package] = {'available': True, 'version': version}
            except ImportError:
                status[package] = {'available': False, 'version': None}
        else:
            status[package] = {'available': False, 'version': None}
    
    return status

