import sys
import importlib.util

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

