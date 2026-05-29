"""
Install SciGraphs addon in Blender from a built extension ZIP file.

Usage:
    blender --background --python install_addon.py -- /path/to/scigraphs-*.zip
"""

import bpy
import sys
import zipfile
import shutil
import platform
from pathlib import Path

# Get ZIP path from command line
if len(sys.argv) < 2 or '--' not in sys.argv:
    print("ERROR: No ZIP file specified")
    print("Usage: blender --background --python install_addon.py -- /path/to/addon.zip")
    sys.exit(1)

# Get the path after '--'
dash_index = sys.argv.index('--')
if dash_index + 1 >= len(sys.argv):
    print("ERROR: No ZIP file specified after '--'")
    sys.exit(1)

zip_path = Path(sys.argv[dash_index + 1])

if not zip_path.exists():
    print(f"ERROR: ZIP file not found: {zip_path}")
    sys.exit(1)

print(f"Installing SciGraphs addon from: {zip_path}")

# Extract addon to extensions directory
extensions_dir = Path(bpy.utils.resource_path('USER')) / 'extensions' / 'user_default'
extensions_dir.mkdir(parents=True, exist_ok=True)
target_dir = extensions_dir / 'scigraphs'

# Remove existing installation
if target_dir.exists():
    print(f"Removing existing installation at: {target_dir}")
    shutil.rmtree(target_dir)

# Extract the ZIP
print("Extracting addon...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(target_dir)

print(f"✓ Extension extracted to: {target_dir}")

# Add extensions directory to sys.path
if str(extensions_dir) not in sys.path:
    sys.path.insert(0, str(extensions_dir))
    print(f"✓ Added to sys.path: {extensions_dir}")

# Unpack wheels to make dependencies available
wheels_dir = target_dir / "wheels"
if wheels_dir.exists():
    print("Unpacking wheel dependencies...")
    site_packages = target_dir / ".site-packages"
    site_packages.mkdir(exist_ok=True)
    
    # Determine platform for wheel filtering
    system = platform.system().lower()
    whl_count = 0
    
    # Unpack all compatible wheels
    for whl_file in wheels_dir.glob("**/*.whl"):
        whl_name = whl_file.name.lower()
        
        # Skip platform-specific wheels that don't match current platform
        if system == "linux":
            if any(p in whl_name for p in ['win_amd64', 'macosx', 'win32']):
                continue
        elif system == "darwin":  # macOS
            if any(p in whl_name for p in ['win_amd64', 'linux', 'win32']):
                continue
        elif system == "windows":
            if any(p in whl_name for p in ['linux', 'macosx']):
                continue
        
        print(f"  Unpacking: {whl_file.name}")
        try:
            with zipfile.ZipFile(whl_file, 'r') as whl:
                whl.extractall(site_packages)
            whl_count += 1
        except Exception as e:
            print(f"  Warning: Failed to unpack {whl_file.name}: {e}")
    
    print(f"✓ Unpacked {whl_count} wheels")
    
    # Add site-packages to sys.path
    if str(site_packages) not in sys.path:
        sys.path.insert(0, str(site_packages))
        print(f"✓ Added to sys.path: {site_packages}")

# Import and register the addon
print("Importing and registering addon...")
try:
    import scigraphs
    print("✓ Addon imported successfully")
    
    # Register the addon
    scigraphs.register()
    print("✓ Addon registered successfully")
    
    # Verify registration
    if hasattr(bpy.types.Scene, 'scigraphs'):
        props = bpy.context.scene.scigraphs
        print("✓ Registration verified - scene properties available")
        print(f"  Available properties: filepath, source_column_index, target_column_index, ...")
    else:
        print("ERROR: Registration failed - scene properties not found")
        sys.exit(1)
    
    # Save preferences to keep addon enabled
    print("Saving preferences to keep addon enabled...")
    bpy.ops.wm.save_userpref()
    print("✓ Preferences saved")
    
    print("\n" + "="*80)
    print("✓ SciGraphs addon installed and registered successfully!")
    print("="*80)
    
except Exception as e:
    print(f"\nERROR: Failed to import/register addon: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

