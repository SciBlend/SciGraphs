# Terrain visualization module for DEM (Digital Elevation Model) import
#
# This module handles:
# - Loading DEM raster files (GeoTIFF, etc.)
# - Fetching DEM data from elevation APIs (Open-Elevation, etc.)
# - Creating terrain mesh in Blender
# - Aligning terrain with OSMnx street network
# - Applying materials and textures

import bpy
import bmesh
import numpy as np
import time
from ...utils.logger import log


def load_dem_data(filepath, bounds=None):
    """
    Load elevation data from a DEM raster file.
    
    Args:
        filepath: Path to DEM file (GeoTIFF, etc.)
        bounds: Optional dict with 'north', 'south', 'east', 'west' to crop
    
    Returns:
        Dict with elevation data, extent, and metadata, or None on error
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
    except ImportError:
        log("rasterio not installed. Install with: pip install rasterio")
        return None
    
    try:
        with rasterio.open(filepath) as src:
            # Get raster metadata
            transform = src.transform
            crs = src.crs
            nodata = src.nodata
            
            # Read full raster or cropped window
            if bounds:
                # Calculate window from bounds
                window = from_bounds(
                    bounds['west'], bounds['south'],
                    bounds['east'], bounds['north'],
                    transform
                )
                elevation = src.read(1, window=window)
                
                # Recalculate transform for the window
                window_transform = src.window_transform(window)
                
                # Calculate actual bounds of the window
                height, width = elevation.shape
                actual_bounds = {
                    'west': window_transform.c,
                    'north': window_transform.f,
                    'east': window_transform.c + width * window_transform.a,
                    'south': window_transform.f + height * window_transform.e,
                }
            else:
                elevation = src.read(1)
                actual_bounds = {
                    'west': src.bounds.left,
                    'east': src.bounds.right,
                    'south': src.bounds.bottom,
                    'north': src.bounds.top,
                }
                window_transform = transform
            
            # Handle nodata values
            if nodata is not None:
                elevation = np.where(elevation == nodata, np.nan, elevation)
            
            # Get resolution
            res_x = abs(window_transform.a) if bounds else abs(transform.a)
            res_y = abs(window_transform.e) if bounds else abs(transform.e)
            
            log(f"DEM loaded: {elevation.shape[1]}x{elevation.shape[0]} pixels")
            log(f"Elevation range: {np.nanmin(elevation):.1f} - {np.nanmax(elevation):.1f} m")
            log(f"Resolution: {res_x:.2f} x {res_y:.2f} degrees/meters")
            
            return {
                'elevation': elevation,
                'bounds': actual_bounds,
                'crs': str(crs) if crs else None,
                'resolution': (res_x, res_y),
                'nodata': nodata,
                'transform': window_transform if bounds else transform,
            }
            
    except Exception as e:
        log(f"Error loading DEM: {e}")
        return None


def fetch_dem_from_api(bounds, resolution=50, api='open-elevation', max_workers=5):
    """
    Fetch elevation data from an online API for a given bounding box.
    
    Args:
        bounds: Dict with 'north', 'south', 'east', 'west' coordinates
        resolution: Number of points per side (total points = resolution^2)
        api: API to use ('open-elevation' or 'opentopodata')
        max_workers: Number of parallel requests to make
    
    Returns:
        Dict with elevation data similar to load_dem_data(), or None on error
    """
    import requests
    
    north = bounds['north']
    south = bounds['south']
    east = bounds['east']
    west = bounds['west']
    
    # Generate grid of lat/lon points
    lats = np.linspace(north, south, resolution)
    lons = np.linspace(west, east, resolution)
    
    # Create list of all coordinates
    coords = []
    for lat in lats:
        for lon in lons:
            coords.append({'latitude': lat, 'longitude': lon})
    
    total_points = len(coords)
    log(f"Fetching elevation for {total_points} points ({resolution}x{resolution} grid) with {max_workers} workers...")
    
    # Query API in batches
    if api == 'open-elevation':
        elevations = _fetch_open_elevation(coords, max_workers=max_workers)
    elif api == 'opentopodata':
        elevations = _fetch_opentopodata(coords, max_workers=max_workers)
    else:
        log(f"Unknown API: {api}")
        return None
    
    if elevations is None or len(elevations) != total_points:
        log("Failed to fetch all elevation data")
        return None
    
    # Reshape to 2D grid
    elevation_grid = np.array(elevations).reshape(resolution, resolution)
    
    # Interpolate NaN values from failed batches
    nan_count = np.sum(np.isnan(elevation_grid))
    if nan_count > 0:
        log(f"Interpolating {nan_count} missing elevation values...")
        elevation_grid = _interpolate_nan_values(elevation_grid)
    
    log(f"DEM from API: {resolution}x{resolution} points")
    log(f"Elevation range: {np.nanmin(elevation_grid):.1f} - {np.nanmax(elevation_grid):.1f} m")
    
    return {
        'elevation': elevation_grid,
        'bounds': bounds,
        'crs': 'EPSG:4326',
        'resolution': ((east - west) / resolution, (north - south) / resolution),
        'nodata': None,
        'transform': None,
        'source': api,
    }


def _fetch_open_elevation(coords, batch_size=100, pause=0.05, max_retries=2, max_workers=5):
    """
    Fetch elevations from Open-Elevation API using parallel requests.
    Free API, no key required.
    Failed batches are marked with NaN and interpolated afterwards.
    
    Args:
        coords: List of coordinate dicts with 'latitude' and 'longitude'
        batch_size: Number of points per API request
        pause: Pause between batch completions (seconds)
        max_retries: Number of retries for failed requests
        max_workers: Number of parallel requests (higher = faster, but may cause rate limiting)
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    url = "https://api.open-elevation.com/api/v1/lookup"
    total = len(coords)
    num_batches = (total + batch_size - 1) // batch_size
    
    # Prepare all batches
    batches = []
    for i in range(0, total, batch_size):
        batch = coords[i:i + batch_size]
        batches.append((i // batch_size, batch))
    
    # Results storage (indexed by batch number)
    results_map = {}
    failed_batches = []
    
    def fetch_batch(batch_info):
        batch_idx, batch = batch_info
        
        for retry in range(max_retries + 1):
            try:
                response = requests.post(url, json={'locations': batch}, timeout=30)
                
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    elevations = []
                    for result in results:
                        elev = result.get('elevation')
                        if elev is not None:
                            elevations.append(float(elev))
                        else:
                            elevations.append(np.nan)
                    return batch_idx, elevations, True
                else:
                    if retry < max_retries:
                        time.sleep(0.3)
                    
            except requests.exceptions.Timeout:
                if retry < max_retries:
                    time.sleep(0.5)
            except Exception:
                if retry < max_retries:
                    time.sleep(0.3)
        
        # All retries failed
        return batch_idx, [np.nan] * len(batch), False
    
    completed = 0
    
    log(f"Fetching {total} points in {num_batches} batches ({max_workers} parallel workers)...")

    executor = ThreadPoolExecutor(max_workers=max_workers)
    futures = {}
    interrupted = False
    try:
        futures = {executor.submit(fetch_batch, b): b[0] for b in batches}

        for future in as_completed(futures):
            batch_idx, elevations, success = future.result()
            results_map[batch_idx] = elevations

            if not success:
                failed_batches.append(batch_idx + 1)

            completed += 1
            progress_pct = 100 * completed / num_batches
            log(f"  Progress: {completed}/{num_batches} batches ({progress_pct:.0f}%)")

            time.sleep(pause)
    except KeyboardInterrupt:
        interrupted = True
        log("Open-Elevation fetch interrupted by user — cancelling pending requests")
        for fut in futures:
            fut.cancel()
    finally:
        # Don't wait for in-flight HTTP requests when the user aborted: that
        # is what made Ctrl+C hang Blender for ~30s in the previous version.
        executor.shutdown(wait=not interrupted, cancel_futures=True)

    if interrupted:
        # Bubble up so the operator can report a clean cancellation.
        raise KeyboardInterrupt("Open-Elevation fetch cancelled")

    all_elevations = []
    for i in range(num_batches):
        all_elevations.extend(results_map.get(i, [np.nan] * batch_size))

    all_elevations = all_elevations[:total]

    if failed_batches:
        log(f"Warning: {len(failed_batches)} batch(es) failed, values will be interpolated")

    return all_elevations


def _fetch_opentopodata(coords, batch_size=100, pause=0.2, max_retries=2, max_workers=2):
    """
    Fetch elevations from OpenTopoData API using parallel requests.
    Free API with multiple datasets (SRTM, ASTER, etc.)
    Failed batches are marked with NaN for later interpolation.
    
    Note: OpenTopoData has stricter rate limits, so fewer workers recommended.
    
    Args:
        coords: List of coordinate dicts with 'latitude' and 'longitude'
        batch_size: Number of points per API request
        pause: Pause between batch completions (seconds)
        max_retries: Number of retries for failed requests
        max_workers: Number of parallel requests (2-3 recommended for this API)
    """
    import requests
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Using SRTM 30m dataset (global coverage)
    url = "https://api.opentopodata.org/v1/srtm30m"
    
    total = len(coords)
    num_batches = (total + batch_size - 1) // batch_size
    
    # Prepare batches
    batches = []
    for i in range(0, total, batch_size):
        batch = coords[i:i + batch_size]
        batches.append((i // batch_size, batch))
    
    results_map = {}
    failed_batches = []
    
    def fetch_batch(batch_info):
        batch_idx, batch = batch_info
        locations = "|".join([f"{c['latitude']},{c['longitude']}" for c in batch])
        
        for retry in range(max_retries + 1):
            try:
                response = requests.get(f"{url}?locations={locations}", timeout=30)
                
                if response.status_code == 200:
                    results = response.json().get('results', [])
                    elevations = []
                    for result in results:
                        elev = result.get('elevation')
                        if elev is not None:
                            elevations.append(float(elev))
                        else:
                            elevations.append(np.nan)
                    return batch_idx, elevations, True
                else:
                    if retry < max_retries:
                        time.sleep(1.0)  # OpenTopoData has stricter rate limit
            
            except Exception:
                if retry < max_retries:
                    time.sleep(1.0)
        
        return batch_idx, [np.nan] * len(batch), False
    
    # Limit workers for OpenTopoData (stricter rate limit)
    effective_workers = min(max_workers, 3)  # Cap at 3 for this API
    completed = 0
    
    log(f"Fetching {total} points in {num_batches} batches ({effective_workers} workers)...")

    executor = ThreadPoolExecutor(max_workers=effective_workers)
    futures = {}
    interrupted = False
    try:
        futures = {executor.submit(fetch_batch, b): b[0] for b in batches}

        for future in as_completed(futures):
            batch_idx, elevations, success = future.result()
            results_map[batch_idx] = elevations

            if not success:
                failed_batches.append(batch_idx + 1)

            completed += 1
            progress_pct = 100 * completed / num_batches
            log(f"  Progress: {completed}/{num_batches} batches ({progress_pct:.0f}%)")

            time.sleep(pause)
    except KeyboardInterrupt:
        interrupted = True
        log("OpenTopoData fetch interrupted by user — cancelling pending requests")
        for fut in futures:
            fut.cancel()
    finally:
        executor.shutdown(wait=not interrupted, cancel_futures=True)

    if interrupted:
        raise KeyboardInterrupt("OpenTopoData fetch cancelled")

    all_elevations = []
    for i in range(num_batches):
        all_elevations.extend(results_map.get(i, [np.nan] * batch_size))

    all_elevations = all_elevations[:total]

    if failed_batches:
        log(f"Warning: {len(failed_batches)} batch(es) failed, values will be interpolated")

    return all_elevations


def _interpolate_nan_values(grid):
    """
    Interpolate NaN values in a 2D grid using nearest neighbor approach.
    This handles gaps from failed API batches.
    """
    from scipy import ndimage
    
    # If no valid data at all, return zeros
    valid_mask = ~np.isnan(grid)
    if not np.any(valid_mask):
        log("Warning: No valid elevation data, using zeros")
        return np.zeros_like(grid)
    
    # Get mean of valid values as fallback
    mean_value = np.nanmean(grid)
    
    # Use distance transform to find nearest valid value indices
    invalid_mask = np.isnan(grid)
    
    # scipy ndimage can fill with nearest neighbor
    indices = ndimage.distance_transform_edt(
        invalid_mask, 
        return_distances=False, 
        return_indices=True
    )
    
    # Create interpolated grid
    result = grid[indices[0], indices[1]]
    
    # Any remaining NaN (shouldn't happen) gets mean value
    result = np.where(np.isnan(result), mean_value, result)
    
    return result


def create_terrain_from_api(bounds, resolution=50, scale=0.001, vertical_scale=1.0,
                            vertical_offset=0.0, api='open-elevation', name="API_Terrain"):
    """
    Create terrain mesh by fetching elevation data from an API.
    
    Args:
        bounds: Dict with 'north', 'south', 'east', 'west'
        resolution: Grid resolution (points per side)
        scale: Scale factor for coordinates
        vertical_scale: Vertical exaggeration
        vertical_offset: Base elevation offset
        api: API to use ('open-elevation' or 'opentopodata')
        name: Name for the Blender object
    
    Returns:
        Created Blender terrain object, or None on error
    """
    # Fetch elevation data
    dem_data = fetch_dem_from_api(bounds, resolution=resolution, api=api)
    
    if dem_data is None:
        return None
    
    # Create mesh from the data
    terrain_obj = create_terrain_mesh(
        dem_data,
        scale=scale,
        vertical_scale=vertical_scale,
        vertical_offset=vertical_offset,
        subsample=1,  # Already at desired resolution
        name=name
    )
    
    if terrain_obj:
        terrain_obj["dem_source"] = api
        terrain_obj["dem_resolution"] = resolution
    
    return terrain_obj


def create_terrain_from_osmnx_api(osmnx_obj, resolution=50, vertical_scale=1.0,
                                   vertical_offset=0.0, api='open-elevation', padding=0.1):
    """
    Create terrain mesh aligned with OSMnx network using elevation API.
    
    Args:
        osmnx_obj: OSMnx graph Blender object
        resolution: Grid resolution (points per side)
        vertical_scale: Vertical exaggeration
        vertical_offset: Base elevation offset
        api: API to use
        padding: Extra padding around network bounds (fraction)
    
    Returns:
        Created terrain object, or None on error
    """
    if osmnx_obj is None or not osmnx_obj.get("is_osmnx", False):
        log("Invalid OSMnx object")
        return None
    
    # Get network extent
    from ..osmnx import analysis as osmnx_analysis
    from ..data_io.importer import _osmnx_graph_cache
    
    G = None
    graph_id = osmnx_obj.get("osmnx_graph_id", "")
    
    if graph_id and graph_id in _osmnx_graph_cache:
        G = _osmnx_graph_cache[graph_id]
    
    if G is None:
        log("Could not find OSMnx graph in cache")
        return None
    
    extent = osmnx_analysis.get_graph_extent(G)
    if extent is None:
        log("Could not get network extent")
        return None
    
    # Add padding
    lat_range = extent['north'] - extent['south']
    lon_range = extent['east'] - extent['west']
    
    bounds = {
        'north': extent['north'] + lat_range * padding,
        'south': extent['south'] - lat_range * padding,
        'east': extent['east'] + lon_range * padding,
        'west': extent['west'] - lon_range * padding,
    }
    
    scale = osmnx_obj.get("osmnx_scale", 0.001)
    
    # Create terrain from API
    terrain_obj = create_terrain_from_api(
        bounds,
        resolution=resolution,
        scale=scale,
        vertical_scale=vertical_scale,
        vertical_offset=vertical_offset,
        api=api,
        name="OSMnx_Terrain_API"
    )
    
    if terrain_obj is None:
        return None
    
    # Apply material
    apply_terrain_material(terrain_obj, style='ELEVATION')
    
    # Link to OSMnx object
    terrain_obj["osmnx_parent"] = osmnx_obj.name
    osmnx_obj["terrain_child"] = terrain_obj.name
    
    # Small z offset to avoid z-fighting
    terrain_obj.location.z = -0.001
    
    log(f"Terrain from API created and aligned with {osmnx_obj.name}")
    
    return terrain_obj


def create_terrain_mesh(dem_data, scale=0.001, vertical_scale=1.0, 
                        vertical_offset=0.0, subsample=1, name="Terrain", 
                        center_lat=None, center_lon=None):
    """
    Create a Blender mesh from DEM elevation data.
    
    Args:
        dem_data: Dict returned by load_dem_data()
        scale: Scale factor to match OSMnx network (meters to Blender units)
        vertical_scale: Vertical exaggeration factor
        vertical_offset: Base elevation offset in meters
        subsample: Subsample factor (1=full resolution, 2=half, etc.)
        name: Name for the Blender object
        center_lat: Latitude of coordinate system center (for alignment with OSMnx)
        center_lon: Longitude of coordinate system center (for alignment with OSMnx)
    
    Returns:
        Created Blender object, or None on error
    """
    if dem_data is None:
        return None
    
    elevation = dem_data['elevation']
    bounds = dem_data['bounds']
    
    # Subsample if requested (for performance with large DEMs)
    if subsample > 1:
        elevation = elevation[::subsample, ::subsample]
    
    height, width = elevation.shape
    log(f"Creating terrain mesh: {width}x{height} vertices")
    
    # Use provided center or calculate from DEM bounds
    if center_lat is None or center_lon is None:
        center_lat = (bounds['north'] + bounds['south']) / 2
        center_lon = (bounds['east'] + bounds['west']) / 2
    
    # Earth parameters for coordinate conversion
    EARTH_RADIUS = 6371000.0
    cos_lat = np.cos(np.radians(center_lat))
    meters_per_deg = np.pi / 180.0 * EARTH_RADIUS
    
    # Calculate vertex positions
    lats = np.linspace(bounds['north'], bounds['south'], height)
    lons = np.linspace(bounds['west'], bounds['east'], width)
    
    # Get min elevation for offset reference
    min_elev = np.nanmin(elevation)
    if np.isnan(min_elev):
        min_elev = 0
    
    # Create mesh
    mesh = bpy.data.meshes.new(name=f"{name}_Mesh")
    bm = bmesh.new()
    
    # Create vertices
    vert_grid = []
    
    for i, lat in enumerate(lats):
        row = []
        for j, lon in enumerate(lons):
            # Convert lat/lon to local meters
            y_m = (lat - center_lat) * meters_per_deg
            x_m = (lon - center_lon) * meters_per_deg * cos_lat
            
            # Get elevation
            elev = elevation[i, j]
            if np.isnan(elev):
                elev = min_elev
            
            # Apply scale and offset
            x = x_m * scale
            y = y_m * scale
            z = ((elev - min_elev) + vertical_offset) * scale * vertical_scale
            
            v = bm.verts.new((x, y, z))
            row.append(v)
        
        vert_grid.append(row)
    
    bm.verts.ensure_lookup_table()
    
    # Create faces (quads)
    for i in range(height - 1):
        for j in range(width - 1):
            v1 = vert_grid[i][j]
            v2 = vert_grid[i][j + 1]
            v3 = vert_grid[i + 1][j + 1]
            v4 = vert_grid[i + 1][j]
            
            try:
                bm.faces.new([v1, v2, v3, v4])
            except ValueError:
                pass  # Skip degenerate faces
    
    # Convert to mesh
    bm.to_mesh(mesh)
    bm.free()
    
    # Create object
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    # Store metadata
    obj["is_terrain"] = True
    obj["dem_bounds_north"] = bounds['north']
    obj["dem_bounds_south"] = bounds['south']
    obj["dem_bounds_east"] = bounds['east']
    obj["dem_bounds_west"] = bounds['west']
    obj["dem_scale"] = scale
    obj["dem_vertical_scale"] = vertical_scale
    obj["dem_center_lat"] = center_lat
    obj["dem_center_lon"] = center_lon
    obj["dem_min_elevation"] = float(min_elev)
    obj["dem_max_elevation"] = float(np.nanmax(elevation))
    
    # Add elevation attribute to vertices
    _add_elevation_attribute(obj, elevation, subsample)
    
    log(f"Terrain mesh created: {len(mesh.vertices)} vertices, {len(mesh.polygons)} faces")
    
    return obj


def _add_elevation_attribute(obj, elevation, subsample):
    """Add elevation as vertex attribute for material use."""
    mesh = obj.data
    height, width = elevation.shape
    
    # Create attribute
    if "elevation" in mesh.attributes:
        mesh.attributes.remove(mesh.attributes["elevation"])
    
    attr = mesh.attributes.new(name="elevation", type='FLOAT', domain='POINT')
    
    # Flatten elevation data to match vertex order
    values = []
    for i in range(height):
        for j in range(width):
            val = elevation[i, j]
            values.append(float(val) if not np.isnan(val) else 0.0)
    
    if len(values) == len(mesh.vertices):
        attr.data.foreach_set("value", values)


def apply_terrain_material(obj, style='ELEVATION'):
    """
    Apply a material to the terrain mesh.
    
    Args:
        obj: Terrain Blender object
        style: Material style ('ELEVATION', 'SIMPLE', 'SATELLITE')
    
    Returns:
        Created material
    """
    if obj is None or not obj.get("is_terrain", False):
        return None
    
    mat_name = f"Terrain_{style}"
    
    # Check if material exists
    mat = bpy.data.materials.get(mat_name)
    if mat is None:
        mat = bpy.data.materials.new(name=mat_name)
        mat.use_nodes = True
        
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        if style == 'ELEVATION':
            _create_elevation_material(nodes, links, obj)
        elif style == 'SIMPLE':
            _create_simple_material(nodes, links)
        else:
            _create_simple_material(nodes, links)
    
    # Assign material to object
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    return mat


def _create_elevation_material(nodes, links, obj):
    """Create a color-ramp material based on elevation."""
    # Output node
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    # Principled BSDF
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (100, 0)
    bsdf.inputs['Roughness'].default_value = 0.8
    
    # Color ramp for elevation
    ramp = nodes.new(type='ShaderNodeValToRGB')
    ramp.location = (-200, 0)
    
    # Set color ramp colors (terrain-like)
    ramp.color_ramp.elements[0].position = 0.0
    ramp.color_ramp.elements[0].color = (0.2, 0.4, 0.1, 1.0)  # Green (low)
    
    ramp.color_ramp.elements[1].position = 1.0
    ramp.color_ramp.elements[1].color = (0.6, 0.5, 0.4, 1.0)  # Brown (high)
    
    # Add intermediate color
    elem = ramp.color_ramp.elements.new(0.5)
    elem.color = (0.5, 0.45, 0.3, 1.0)  # Tan (middle)
    
    # Map range to normalize elevation
    map_range = nodes.new(type='ShaderNodeMapRange')
    map_range.location = (-400, 0)
    
    min_elev = obj.get("dem_min_elevation", 0)
    max_elev = obj.get("dem_max_elevation", 1000)
    
    map_range.inputs['From Min'].default_value = min_elev
    map_range.inputs['From Max'].default_value = max_elev
    map_range.inputs['To Min'].default_value = 0.0
    map_range.inputs['To Max'].default_value = 1.0
    
    # Attribute node to read elevation
    attr_node = nodes.new(type='ShaderNodeAttribute')
    attr_node.location = (-600, 0)
    attr_node.attribute_name = "elevation"
    
    # Link nodes
    links.new(attr_node.outputs['Fac'], map_range.inputs['Value'])
    links.new(map_range.outputs['Result'], ramp.inputs['Fac'])
    links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])


def _create_simple_material(nodes, links):
    """Create a simple solid color material."""
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (100, 0)
    bsdf.inputs['Base Color'].default_value = (0.3, 0.35, 0.25, 1.0)
    bsdf.inputs['Roughness'].default_value = 0.9
    
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])


def create_terrain_from_osmnx(osmnx_obj, dem_filepath, vertical_scale=1.0, 
                               vertical_offset=0.0, subsample=1, padding=0.1):
    """
    Create terrain mesh aligned with an OSMnx street network.
    
    Args:
        osmnx_obj: OSMnx graph Blender object
        dem_filepath: Path to DEM file
        vertical_scale: Vertical exaggeration
        vertical_offset: Base elevation offset
        subsample: Subsample factor for large DEMs
        padding: Extra padding around network bounds (fraction)
    
    Returns:
        Created terrain object, or None on error
    """
    if osmnx_obj is None or not osmnx_obj.get("is_osmnx", False):
        log("Invalid OSMnx object")
        return None
    
    # Get network extent from mesh vertices
    mesh = osmnx_obj.data
    scale = osmnx_obj.get("osmnx_scale", 0.001)
    
    if len(mesh.vertices) == 0:
        log("OSMnx mesh has no vertices")
        return None
    
    # Calculate bounds from vertices
    xs = [v.co.x for v in mesh.vertices]
    ys = [v.co.y for v in mesh.vertices]
    
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Convert back to lat/lon (reverse the OSMnx coordinate transformation)
    # This is approximate but should work for the DEM cropping
    
    # Get stored center if available, otherwise estimate
    from ..osmnx import analysis as osmnx_analysis
    from ..data_io.importer import _osmnx_graph_cache
    
    G = None
    graph_id = osmnx_obj.get("osmnx_graph_id", "")
    if graph_id and hasattr(osmnx_analysis, 'get_graph_extent'):
        # Try to get graph from cache
        if graph_id in _osmnx_graph_cache:
            G = _osmnx_graph_cache[graph_id]
    
    if G is not None:
        extent = osmnx_analysis.get_graph_extent(G)
        if extent:
            bounds = {
                'north': extent['north'],
                'south': extent['south'],
                'east': extent['east'],
                'west': extent['west'],
            }
            
            # Add padding
            lat_range = bounds['north'] - bounds['south']
            lon_range = bounds['east'] - bounds['west']
            
            bounds['north'] += lat_range * padding
            bounds['south'] -= lat_range * padding
            bounds['east'] += lon_range * padding
            bounds['west'] -= lon_range * padding
        else:
            bounds = None
    else:
        bounds = None
        log("Could not get network extent, loading full DEM")
    
    # Load DEM data
    dem_data = load_dem_data(dem_filepath, bounds=bounds)
    
    if dem_data is None:
        return None
    
    # Get center coordinates from OSMnx object for alignment
    center_lat = osmnx_obj.get("osmnx_center_lat")
    center_lon = osmnx_obj.get("osmnx_center_lon")
    
    # Create terrain mesh with same scale and center as OSMnx network
    terrain_obj = create_terrain_mesh(
        dem_data,
        scale=scale,
        vertical_scale=vertical_scale,
        vertical_offset=vertical_offset,
        subsample=subsample,
        name="OSMnx_Terrain",
        center_lat=center_lat,
        center_lon=center_lon
    )
    
    if terrain_obj is None:
        return None
    
    # Apply elevation material
    apply_terrain_material(terrain_obj, style='ELEVATION')
    
    # Link terrain to OSMnx object
    terrain_obj["osmnx_parent"] = osmnx_obj.name
    osmnx_obj["terrain_child"] = terrain_obj.name
    
    # Position terrain slightly below the network
    terrain_obj.location.z = -0.001  # Small offset to avoid z-fighting
    
    log(f"Terrain created and aligned with {osmnx_obj.name}")
    
    return terrain_obj


def update_terrain_vertical_scale(terrain_obj, vertical_scale):
    """
    Update the vertical scale of an existing terrain mesh.
    
    Args:
        terrain_obj: Terrain Blender object
        vertical_scale: New vertical scale factor
    """
    if terrain_obj is None or not terrain_obj.get("is_terrain", False):
        return
    
    old_scale = terrain_obj.get("dem_vertical_scale", 1.0)
    
    if old_scale == vertical_scale:
        return
    
    # Calculate scale factor
    factor = vertical_scale / old_scale
    
    # Scale Z coordinates of all vertices
    mesh = terrain_obj.data
    for v in mesh.vertices:
        v.co.z *= factor
    
    mesh.update()
    
    # Update stored scale
    terrain_obj["dem_vertical_scale"] = vertical_scale
    
    log(f"Terrain vertical scale updated to {vertical_scale}")


def remove_terrain(terrain_obj):
    """
    Remove a terrain object and clean up references.
    
    Args:
        terrain_obj: Terrain Blender object to remove
    """
    if terrain_obj is None:
        return
    
    # Remove reference from parent OSMnx object
    parent_name = terrain_obj.get("osmnx_parent", "")
    if parent_name and parent_name in bpy.data.objects:
        parent = bpy.data.objects[parent_name]
        if "terrain_child" in parent:
            del parent["terrain_child"]
    
    # Remove mesh data
    mesh = terrain_obj.data
    
    # Remove object
    bpy.data.objects.remove(terrain_obj, do_unlink=True)
    
    # Remove mesh if orphaned
    if mesh.users == 0:
        bpy.data.meshes.remove(mesh)
    
    log("Terrain removed")


def apply_dem_elevations_to_graph(osmnx_obj, dem_data, vertical_scale=1.0, vertical_offset=0.0):
    """
    Apply elevation data from DEM to the OSMnx graph vertices.
    This ensures the graph and terrain use the same elevation source.
    
    Args:
        osmnx_obj: OSMnx Blender object
        dem_data: Dict with elevation data from load_dem_data() or fetch_dem_from_api()
        vertical_scale: Vertical exaggeration factor
        vertical_offset: Base elevation offset in meters
    
    Returns:
        True on success, False on error
    """
    if osmnx_obj is None or dem_data is None:
        return False

    mesh = osmnx_obj.data
    # SciGraphs has historically stored the network scale under two custom
    # property names depending on which import path created the object
    # (``"scale"`` for the main downloader, ``"osmnx_scale"`` for the
    # cache/IO/data operators). Read both so elevations map correctly
    # regardless of which path was used.
    scale = osmnx_obj.get("osmnx_scale")
    if scale is None:
        scale = osmnx_obj.get("scale", 0.001)

    elevation_grid = dem_data['elevation']
    bounds = dem_data['bounds']

    min_elev = np.nanmin(elevation_grid)
    max_elev = np.nanmax(elevation_grid)
    if np.isnan(min_elev):
        min_elev = 0
    if np.isnan(max_elev):
        max_elev = min_elev

    # Single source of truth for the projection: the addon stores the
    # equirectangular-local origin used to build the mesh on the object
    # itself. Sampling the DEM through the *mesh* coordinates (instead of
    # the cached MultiDiGraph node coordinates) avoids the entire family
    # of bugs that arise when the cached graph and the visible mesh
    # diverge — e.g. simplified / projected / converted graphs whose
    # node IDs no longer match ``nodes_data`` 1:1, or projected graphs
    # where ``G.nodes[n]['x','y']`` are UTM metres instead of degrees.
    EARTH_RADIUS = 6_371_000.0
    center_lat = osmnx_obj.get("osmnx_center_lat")
    center_lon = osmnx_obj.get("osmnx_center_lon")
    if center_lat is None or center_lon is None:
        # Legacy objects: fall back to the DEM bbox centre. Less accurate
        # if the graph was downloaded with a different padding, but
        # better than refusing to apply elevations.
        center_lat = (bounds['north'] + bounds['south']) / 2
        center_lon = (bounds['east'] + bounds['west']) / 2
        log("Network has no osmnx_center_lat/lon; using DEM bbox centre as fallback")
    center_lat = float(center_lat)
    center_lon = float(center_lon)

    cos_lat = np.cos(np.radians(center_lat))
    if abs(cos_lat) < 1e-9:
        cos_lat = 1.0  # safety at the poles
    inv_scale = 1.0 / float(scale) if scale else 1.0

    # Sample elevation per mesh vertex by inverting the equirectangular
    # projection that originally placed the vertex. This makes the
    # function robust to any mismatch between the cached graph and the
    # mesh (simplification, conversion, projection) because the mesh is
    # the only thing being rendered anyway.
    vertex_elevations = [0.0] * len(mesh.vertices)
    out_of_bounds = 0
    for vert_idx, vert in enumerate(mesh.vertices):
        x_m = vert.co.x * inv_scale
        y_m = vert.co.y * inv_scale
        lat = center_lat + (y_m / (np.pi / 180.0 * EARTH_RADIUS))
        lon = center_lon + (x_m / (np.pi / 180.0 * EARTH_RADIUS * cos_lat))

        # Track vertices whose true geographic position is outside the
        # DEM tile. ``_sample_elevation_from_grid`` clamps to the border
        # so we still get a sane value, but having lots of these usually
        # means the user fetched a too-small DEM and should re-run with
        # more padding.
        if not (bounds['south'] <= lat <= bounds['north']
                and bounds['west'] <= lon <= bounds['east']):
            out_of_bounds += 1

        elev = _sample_elevation_from_grid(lat, lon, elevation_grid, bounds)
        if np.isnan(elev):
            elev = min_elev

        vertex_elevations[vert_idx] = elev
        z = ((elev - min_elev) + vertical_offset) * scale * vertical_scale
        vert.co.z = z

    if out_of_bounds:
        pct = 100.0 * out_of_bounds / max(len(mesh.vertices), 1)
        log(
            f"  Warning: {out_of_bounds} ({pct:.1f}%) of vertices lie outside "
            "the DEM tile and were clamped to the nearest edge. "
            "Increase 'Padding' before fetching the DEM if this is large."
        )

    # Mirror the per-vertex elevations back into the cached graph so
    # downstream code (grade calculations, GraphML export, etc.) can
    # still use ``G.nodes[n]['elevation']`` as before.
    nodes_str = osmnx_obj.get("nodes_data", "")
    node_ids = nodes_str.split(",") if nodes_str else []
    from ..data_io.importer import _osmnx_graph_cache
    graph_id = osmnx_obj.get("osmnx_graph_id", "")
    G = _osmnx_graph_cache.get(graph_id) if graph_id else None
    if G is not None and node_ids:
        for i, node_id_str in enumerate(node_ids):
            if i >= len(vertex_elevations):
                break
            try:
                node_id = int(node_id_str) if node_id_str.lstrip('-').isdigit() else node_id_str
            except ValueError:
                node_id = node_id_str
            if node_id in G.nodes:
                G.nodes[node_id]['elevation'] = float(vertex_elevations[i])

    mesh.update()
    
    # Create elevation attribute
    attr_name = "elevation"
    if attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name])
    
    elev_attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
    elev_attr.data.foreach_set("value", vertex_elevations)
    
    # Update object properties
    osmnx_obj["osmnx_has_elevation"] = True
    osmnx_obj["osmnx_3d_applied"] = True
    osmnx_obj["osmnx_elev_scale_used"] = vertical_scale
    osmnx_obj["osmnx_elev_min"] = float(min_elev)
    osmnx_obj["osmnx_elev_max"] = float(max_elev)
    osmnx_obj["osmnx_elev_range"] = float(max_elev - min_elev)
    
    log(f"Applied DEM elevations to {len(mesh.vertices)} vertices (range: {max_elev - min_elev:.1f}m)")
    
    return True


def _sample_elevation_from_grid(lat, lon, elevation_grid, bounds):
    """
    Sample elevation value from grid at a given lat/lon coordinate.
    Uses bilinear interpolation.
    """
    height, width = elevation_grid.shape
    
    # Calculate pixel coordinates
    x_frac = (lon - bounds['west']) / (bounds['east'] - bounds['west'])
    y_frac = (bounds['north'] - lat) / (bounds['north'] - bounds['south'])
    
    # Clamp to valid range
    x_frac = max(0, min(1, x_frac))
    y_frac = max(0, min(1, y_frac))
    
    # Convert to pixel indices
    x = x_frac * (width - 1)
    y = y_frac * (height - 1)
    
    # Bilinear interpolation
    x0 = int(x)
    y0 = int(y)
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    
    dx = x - x0
    dy = y - y0
    
    # Get four corner values
    v00 = elevation_grid[y0, x0]
    v10 = elevation_grid[y0, x1]
    v01 = elevation_grid[y1, x0]
    v11 = elevation_grid[y1, x1]
    
    # Handle NaN values
    values = [v00, v10, v01, v11]
    valid_values = [v for v in values if not np.isnan(v)]
    
    if not valid_values:
        return 0.0
    
    # Replace NaN with mean of valid values
    mean_val = np.mean(valid_values)
    v00 = v00 if not np.isnan(v00) else mean_val
    v10 = v10 if not np.isnan(v10) else mean_val
    v01 = v01 if not np.isnan(v01) else mean_val
    v11 = v11 if not np.isnan(v11) else mean_val
    
    # Interpolate
    elev = (v00 * (1 - dx) * (1 - dy) +
            v10 * dx * (1 - dy) +
            v01 * (1 - dx) * dy +
            v11 * dx * dy)
    
    return float(elev)


def import_dem_unified(osmnx_obj, dem_source, source_type='api', resolution=50,
                       vertical_scale=1.0, vertical_offset=0.0, show_terrain=True,
                       api='open-elevation', subsample=1, padding=0.1, max_workers=5):
    """
    Unified DEM import that creates terrain AND applies elevations to graph.
    
    Args:
        osmnx_obj: OSMnx Blender object
        dem_source: File path (if source_type='file') or ignored (if 'api')
        source_type: 'file' or 'api'
        resolution: Grid resolution for API (points per side)
        vertical_scale: Vertical exaggeration
        vertical_offset: Base elevation offset
        show_terrain: Whether to create visible terrain mesh
        api: API to use if source_type='api'
        subsample: Subsample factor for file source
        padding: Padding around network bounds
    
    Returns:
        Tuple (terrain_obj or None, success_bool)
    """
    if osmnx_obj is None or not osmnx_obj.get("is_osmnx", False):
        log("Invalid OSMnx object")
        return None, False
    
    scale = osmnx_obj.get("osmnx_scale", 0.001)
    
    # Get network bounds
    from ..osmnx import analysis as osmnx_analysis
    from ..data_io.importer import _osmnx_graph_cache
    
    G = None
    graph_id = osmnx_obj.get("osmnx_graph_id", "")
    if graph_id and graph_id in _osmnx_graph_cache:
        G = _osmnx_graph_cache[graph_id]
    
    if G is None:
        log("Could not find OSMnx graph in cache")
        return None, False
    
    extent = osmnx_analysis.get_graph_extent(G)
    if extent is None:
        log("Could not get network extent")
        return None, False
    
    # Add padding to bounds
    lat_range = extent['north'] - extent['south']
    lon_range = extent['east'] - extent['west']
    
    bounds = {
        'north': extent['north'] + lat_range * padding,
        'south': extent['south'] - lat_range * padding,
        'east': extent['east'] + lon_range * padding,
        'west': extent['west'] - lon_range * padding,
    }
    
    # Get DEM data
    if source_type == 'api':
        dem_data = fetch_dem_from_api(bounds, resolution=resolution, api=api, max_workers=max_workers)
    else:
        dem_data = load_dem_data(dem_source, bounds=bounds)
    
    if dem_data is None:
        log("Failed to get DEM data")
        return None, False
    
    # Apply elevations to graph
    success = apply_dem_elevations_to_graph(
        osmnx_obj, dem_data,
        vertical_scale=vertical_scale,
        vertical_offset=vertical_offset
    )
    
    if not success:
        log("Failed to apply elevations to graph")
        return None, False
    
    terrain_obj = None
    
    # Create terrain mesh if requested
    if show_terrain:
        terrain_obj = create_terrain_mesh(
            dem_data,
            scale=scale,
            vertical_scale=vertical_scale,
            vertical_offset=vertical_offset,
            subsample=subsample if source_type == 'file' else 1,
            name="OSMnx_Terrain"
        )
        
        if terrain_obj:
            apply_terrain_material(terrain_obj, style='ELEVATION')
            
            terrain_obj["osmnx_parent"] = osmnx_obj.name
            terrain_obj["dem_source"] = api if source_type == 'api' else 'file'
            osmnx_obj["terrain_child"] = terrain_obj.name
            
            # Position slightly below to avoid z-fighting
            terrain_obj.location.z = -0.001
    
    log(f"DEM import complete. Terrain: {'created' if terrain_obj else 'hidden'}")
    
    return terrain_obj, True


def toggle_terrain_visibility(osmnx_obj, visible):
    """
    Toggle visibility of terrain associated with an OSMnx graph.
    
    Args:
        osmnx_obj: OSMnx Blender object
        visible: True to show, False to hide
    """
    if osmnx_obj is None:
        return
    
    terrain_name = osmnx_obj.get("terrain_child", "")
    if terrain_name and terrain_name in bpy.data.objects:
        terrain_obj = bpy.data.objects[terrain_name]
        terrain_obj.hide_viewport = not visible
        terrain_obj.hide_render = not visible


def get_osmnx_bounds(osmnx_obj, padding=0.1):
    """
    Get the geographic bounds of an OSMnx object in WGS84 (EPSG:4326).
    
    Args:
        osmnx_obj: OSMnx Blender object
        padding: Extra padding as fraction of extent (0.1 = 10%)
    
    Returns:
        Dict with 'north', 'south', 'east', 'west' in degrees (WGS84) or None
    """
    if osmnx_obj is None or not osmnx_obj.get("is_osmnx", False):
        return None
    
    from ..data_io.importer import _osmnx_graph_cache
    
    graph_id = osmnx_obj.get("osmnx_graph_id", "")
    G = _osmnx_graph_cache.get(graph_id) if graph_id else None
    
    if G is None:
        log("Could not find OSMnx graph in cache")
        return None
    
    # Get coordinates from graph, handling projected CRS
    extent = _get_wgs84_extent(G)
    if extent is None:
        return None
    
    # Add padding
    lat_range = extent['north'] - extent['south']
    lon_range = extent['east'] - extent['west']
    
    bounds = {
        'north': extent['north'] + lat_range * padding,
        'south': extent['south'] - lat_range * padding,
        'east': extent['east'] + lon_range * padding,
        'west': extent['west'] - lon_range * padding,
    }
    
    return bounds


def _get_wgs84_extent(G):
    """
    Get graph extent in WGS84 coordinates, reprojecting if necessary.
    
    Args:
        G: OSMnx graph (may be projected or unprojected)
    
    Returns:
        Dict with 'north', 'south', 'east', 'west' in WGS84 degrees
    """
    if G is None:
        return None
    
    try:
        import osmnx as ox
        
        # Check if graph is projected (not in lat/lon)
        crs = G.graph.get('crs', None)
        is_projected = False
        
        if crs is not None:
            # CRS can be a string like 'EPSG:4326' or a pyproj CRS object
            crs_str = str(crs).upper()
            # EPSG:4326 is WGS84 (unprojected)
            if 'EPSG:4326' not in crs_str and 'WGS 84' not in crs_str:
                is_projected = True
        
        # If projected, we need to reproject to WGS84
        if is_projected:
            log(f"Graph is projected ({crs}), reprojecting to WGS84...")
            G_wgs84 = ox.project_graph(G, to_crs='EPSG:4326')
        else:
            G_wgs84 = G
        
        # Extract coordinates
        lats = []
        lons = []
        for node, data in G_wgs84.nodes(data=True):
            if 'y' in data and 'x' in data:
                lat = data['y']
                lon = data['x']
                
                # Sanity check: valid lat/lon ranges
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    lats.append(lat)
                    lons.append(lon)
        
        if not lats or not lons:
            log("No valid WGS84 coordinates found in graph")
            return None
        
        return {
            'north': max(lats),
            'south': min(lats),
            'east': max(lons),
            'west': min(lons),
        }
        
    except Exception as e:
        log(f"Error getting WGS84 extent: {e}")
        return None


def export_bounds_geojson(bounds, filepath):
    """
    Export bounds as GeoJSON polygon file.
    Compatible with Copernicus Data Space (EPSG:4326 required).
    
    Args:
        bounds: Dict with 'north', 'south', 'east', 'west' in WGS84 degrees
        filepath: Output file path (.geojson)
    
    Returns:
        True on success, False on error
    """
    import json
    
    if bounds is None:
        return False
    
    # Validate that coordinates are in valid WGS84 range
    if not (-90 <= bounds['south'] <= 90 and -90 <= bounds['north'] <= 90):
        log(f"Invalid latitude values: {bounds['south']}, {bounds['north']}")
        log("Coordinates must be in WGS84 (EPSG:4326)")
        return False
    
    if not (-180 <= bounds['west'] <= 180 and -180 <= bounds['east'] <= 180):
        log(f"Invalid longitude values: {bounds['west']}, {bounds['east']}")
        log("Coordinates must be in WGS84 (EPSG:4326)")
        return False
    
    # Create polygon coordinates (closed ring)
    # Format: [longitude, latitude] as per GeoJSON spec
    coordinates = [[
        [bounds['west'], bounds['south']],
        [bounds['east'], bounds['south']],
        [bounds['east'], bounds['north']],
        [bounds['west'], bounds['north']],
        [bounds['west'], bounds['south']],  # Close the ring
    ]]
    
    # Simple GeoJSON that Copernicus accepts (minimal properties)
    geojson = {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": coordinates
            }
        }]
    }
    
    try:
        with open(filepath, 'w') as f:
            json.dump(geojson, f, indent=2)
        log(f"GeoJSON exported: {filepath}")
        return True
    except Exception as e:
        log(f"Error exporting GeoJSON: {e}")
        return False


def export_bounds_kml(bounds, filepath):
    """
    Export bounds as KML file.
    Compatible with Copernicus Data Space and Google Earth.
    
    Args:
        bounds: Dict with 'north', 'south', 'east', 'west'
        filepath: Output file path (.kml)
    
    Returns:
        True on success, False on error
    """
    if bounds is None:
        return False
    
    # KML coordinates are lon,lat,alt (space separated, comma between points)
    coords = (
        f"{bounds['west']},{bounds['south']},0 "
        f"{bounds['east']},{bounds['south']},0 "
        f"{bounds['east']},{bounds['north']},0 "
        f"{bounds['west']},{bounds['north']},0 "
        f"{bounds['west']},{bounds['south']},0"
    )
    
    kml_content = f'''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <name>SciGraphs Area of Interest</name>
    <description>Exported from SciGraphs Blender addon for DEM download</description>
    <Style id="aoi_style">
      <LineStyle>
        <color>ff0000ff</color>
        <width>2</width>
      </LineStyle>
      <PolyStyle>
        <color>4d0000ff</color>
      </PolyStyle>
    </Style>
    <Placemark>
      <name>Area of Interest</name>
      <description>
        North: {bounds['north']:.6f}
        South: {bounds['south']:.6f}
        East: {bounds['east']:.6f}
        West: {bounds['west']:.6f}
      </description>
      <styleUrl>#aoi_style</styleUrl>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>{coords}</coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>'''
    
    try:
        with open(filepath, 'w') as f:
            f.write(kml_content)
        log(f"KML exported: {filepath}")
        return True
    except Exception as e:
        log(f"Error exporting KML: {e}")
        return False


def export_bounds_wkt(bounds, filepath):
    """
    Export bounds as WKT (Well-Known Text) file.
    Compatible with Copernicus Data Space.
    
    Args:
        bounds: Dict with 'north', 'south', 'east', 'west'
        filepath: Output file path (.wkt)
    
    Returns:
        True on success, False on error
    """
    if bounds is None:
        return False
    
    # WKT polygon format
    wkt = (
        f"POLYGON(("
        f"{bounds['west']} {bounds['south']}, "
        f"{bounds['east']} {bounds['south']}, "
        f"{bounds['east']} {bounds['north']}, "
        f"{bounds['west']} {bounds['north']}, "
        f"{bounds['west']} {bounds['south']}"
        f"))"
    )
    
    try:
        with open(filepath, 'w') as f:
            f.write(wkt)
        log(f"WKT exported: {filepath}")
        return True
    except Exception as e:
        log(f"Error exporting WKT: {e}")
        return False


def export_aoi_for_copernicus(osmnx_obj, filepath, format='geojson', padding=0.1):
    """
    Export the area of interest for use with Copernicus Data Space.
    
    Args:
        osmnx_obj: OSMnx Blender object
        filepath: Output file path
        format: Export format ('geojson', 'kml', or 'wkt')
        padding: Extra padding as fraction of extent
    
    Returns:
        Tuple (success, bounds_info_dict)
    """
    bounds = get_osmnx_bounds(osmnx_obj, padding=padding)
    
    if bounds is None:
        return False, None
    
    # Calculate area info
    lat_mid = (bounds['north'] + bounds['south']) / 2
    lat_km = 111.0
    lon_km = 111.0 * np.cos(np.radians(lat_mid))
    
    width_km = (bounds['east'] - bounds['west']) * lon_km
    height_km = (bounds['north'] - bounds['south']) * lat_km
    area_km2 = width_km * height_km
    
    bounds_info = {
        'north': bounds['north'],
        'south': bounds['south'],
        'east': bounds['east'],
        'west': bounds['west'],
        'width_km': width_km,
        'height_km': height_km,
        'area_km2': area_km2,
    }
    
    # Export in requested format
    if format == 'geojson':
        success = export_bounds_geojson(bounds, filepath)
    elif format == 'kml':
        success = export_bounds_kml(bounds, filepath)
    elif format == 'wkt':
        success = export_bounds_wkt(bounds, filepath)
    else:
        log(f"Unknown format: {format}")
        return False, bounds_info
    
    return success, bounds_info


# =============================================================================
# TERRAIN PLANE IMPORT (Textured plane from raster/KMZ)
# =============================================================================

# Supported CRS definitions
SUPPORTED_CRS = {
    'EPSG:4326': {
        'name': 'WGS 84',
        'description': 'World Geodetic System 1984 (GPS coordinates)',
        'is_geographic': True,
    },
    'EPSG:3857': {
        'name': 'Web Mercator',
        'description': 'Used by Google Maps, OpenStreetMap',
        'is_geographic': False,
    },
    'EPSG:32630': {
        'name': 'UTM Zone 30N',
        'description': 'Universal Transverse Mercator Zone 30 North (Spain, UK)',
        'is_geographic': False,
    },
}


def import_terrain_plane(filepath, source_crs='EPSG:4326', target_crs='EPSG:4326',
                         osmnx_obj=None, name="Terrain_Plane"):
    """
    Import a raster file as a textured terrain plane.
    
    Supports: GeoTIFF (8/16/32 bit), KMZ with embedded images, PNG, JPG
    
    Args:
        filepath: Path to raster file
        source_crs: CRS of the input file
        target_crs: Target CRS for the mesh
        osmnx_obj: Optional OSMnx object to align with
        name: Name for the Blender object
    
    Returns:
        Tuple (terrain_object, metadata_dict) or (None, None) on error
    """
    import os
    
    ext = os.path.splitext(filepath)[1].lower()
    
    # Route to appropriate importer
    if ext == '.kmz':
        return _import_kmz_terrain(filepath, source_crs, target_crs, osmnx_obj, name)
    elif ext in ['.tif', '.tiff', '.geotiff']:
        return _import_geotiff_terrain(filepath, source_crs, target_crs, osmnx_obj, name)
    elif ext in ['.png', '.jpg', '.jpeg']:
        return _import_image_terrain(filepath, source_crs, target_crs, osmnx_obj, name)
    else:
        log(f"Unsupported file format: {ext}")
        return None, None


def _import_kmz_terrain(filepath, source_crs, target_crs, osmnx_obj, name):
    """Import KMZ file (Google Earth format with embedded image)."""
    import zipfile
    import tempfile
    import os
    import xml.etree.ElementTree as ET
    
    try:
        with zipfile.ZipFile(filepath, 'r') as kmz:
            # List contents
            contents = kmz.namelist()
            log(f"KMZ contents: {contents}")
            
            # Find KML file
            kml_file = None
            for f in contents:
                if f.lower().endswith('.kml'):
                    kml_file = f
                    break
            
            if not kml_file:
                log("No KML file found in KMZ")
                return None, None
            
            # Extract to temp dir
            temp_dir = tempfile.mkdtemp()
            kmz.extractall(temp_dir)
            
            # Parse KML for bounds and image reference
            kml_path = os.path.join(temp_dir, kml_file)
            bounds, image_path = _parse_kml_ground_overlay(kml_path, temp_dir)
            
            if bounds is None or image_path is None:
                log("Could not parse KML ground overlay")
                return None, None
            
            log(f"KMZ bounds: N={bounds['north']:.5f}, S={bounds['south']:.5f}, "
                f"E={bounds['east']:.5f}, W={bounds['west']:.5f}")
            log(f"Image: {image_path}")
            
            # Create terrain plane with the image
            return _create_textured_terrain_plane(
                image_path, bounds, source_crs, target_crs, osmnx_obj, name
            )
            
    except Exception as e:
        log(f"Error importing KMZ: {e}")
        return None, None


def _parse_kml_ground_overlay(kml_path, base_dir):
    """Parse KML file to extract GroundOverlay bounds and image path."""
    import xml.etree.ElementTree as ET
    import os
    
    try:
        tree = ET.parse(kml_path)
        root = tree.getroot()
        
        # Handle KML namespace
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}
        
        # Try with namespace first, then without
        ground_overlay = root.find('.//kml:GroundOverlay', ns)
        if ground_overlay is None:
            ground_overlay = root.find('.//GroundOverlay')
        
        if ground_overlay is None:
            # Try finding any element with LatLonBox
            for elem in root.iter():
                if 'LatLonBox' in elem.tag or elem.find('.//LatLonBox') is not None:
                    ground_overlay = elem
                    break
        
        if ground_overlay is None:
            log("No GroundOverlay found in KML")
            return None, None
        
        # Get bounds from LatLonBox
        lat_lon_box = ground_overlay.find('.//kml:LatLonBox', ns)
        if lat_lon_box is None:
            lat_lon_box = ground_overlay.find('.//LatLonBox')
        
        if lat_lon_box is None:
            log("No LatLonBox found")
            return None, None
        
        def get_value(parent, tag, ns):
            elem = parent.find(f'.//kml:{tag}', ns)
            if elem is None:
                elem = parent.find(f'.//{tag}')
            return float(elem.text) if elem is not None else None
        
        bounds = {
            'north': get_value(lat_lon_box, 'north', ns),
            'south': get_value(lat_lon_box, 'south', ns),
            'east': get_value(lat_lon_box, 'east', ns),
            'west': get_value(lat_lon_box, 'west', ns),
        }
        
        if any(v is None for v in bounds.values()):
            log(f"Incomplete bounds: {bounds}")
            return None, None
        
        # Get image path from Icon/href
        icon = ground_overlay.find('.//kml:Icon', ns)
        if icon is None:
            icon = ground_overlay.find('.//Icon')
        
        if icon is not None:
            href = icon.find('.//kml:href', ns)
            if href is None:
                href = icon.find('.//href')
            
            if href is not None and href.text:
                image_path = os.path.join(base_dir, href.text)
                if os.path.exists(image_path):
                    return bounds, image_path
        
        # Try to find any image file in the directory
        for f in os.listdir(base_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                return bounds, os.path.join(base_dir, f)
        
        return bounds, None
        
    except Exception as e:
        log(f"Error parsing KML: {e}")
        return None, None


def _import_geotiff_terrain(filepath, source_crs, target_crs, osmnx_obj, name):
    """Import GeoTIFF as terrain plane (supports 8, 16, 32 bit)."""
    try:
        import rasterio
        from rasterio.warp import calculate_default_transform, reproject, Resampling
    except ImportError:
        log("rasterio not installed")
        return None, None
    
    try:
        with rasterio.open(filepath) as src:
            # Get file info
            bit_depth = src.dtypes[0]
            log(f"GeoTIFF: {src.width}x{src.height}, {bit_depth}, {len(src.indexes)} band(s)")
            
            # Get bounds
            bounds = {
                'west': src.bounds.left,
                'south': src.bounds.bottom,
                'east': src.bounds.right,
                'north': src.bounds.top,
            }
            
            file_crs = str(src.crs) if src.crs else source_crs
            log(f"File CRS: {file_crs}")
            
            # Reproject bounds if needed
            if file_crs != 'EPSG:4326' and source_crs != file_crs:
                log(f"Using specified CRS: {source_crs}")
                file_crs = source_crs
            
            # Convert bounds to WGS84 for alignment
            bounds_wgs84 = _reproject_bounds(bounds, file_crs, 'EPSG:4326')
            if bounds_wgs84 is None:
                bounds_wgs84 = bounds
            
            log(f"Bounds (WGS84): N={bounds_wgs84['north']:.5f}, S={bounds_wgs84['south']:.5f}")
            
            # Read image data
            if src.count >= 3:
                # RGB image
                img_data = src.read([1, 2, 3])
            else:
                # Single band (grayscale or DEM)
                img_data = src.read(1)
            
            # Normalize to 0-255 for texture
            img_normalized = _normalize_raster_data(img_data, bit_depth)
            
            # Save as temp PNG for Blender
            import tempfile
            import os
            from PIL import Image
            
            temp_dir = tempfile.mkdtemp()
            temp_image = os.path.join(temp_dir, "terrain_texture.png")
            
            if len(img_normalized.shape) == 3:
                # RGB
                img_pil = Image.fromarray(np.transpose(img_normalized, (1, 2, 0)).astype(np.uint8))
            else:
                # Grayscale
                img_pil = Image.fromarray(img_normalized.astype(np.uint8))
            
            img_pil.save(temp_image)
            
            return _create_textured_terrain_plane(
                temp_image, bounds_wgs84, 'EPSG:4326', target_crs, osmnx_obj, name
            )
            
    except Exception as e:
        log(f"Error importing GeoTIFF: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def _import_image_terrain(filepath, source_crs, target_crs, osmnx_obj, name):
    """Import regular image as terrain plane (needs manual bounds or OSMnx alignment)."""
    
    if osmnx_obj is None:
        log("Image import requires an OSMnx object for georeferencing")
        return None, None
    
    # Get bounds from OSMnx object
    bounds = get_osmnx_bounds(osmnx_obj, padding=0.1)
    if bounds is None:
        log("Could not get bounds from OSMnx object")
        return None, None
    
    return _create_textured_terrain_plane(
        filepath, bounds, 'EPSG:4326', target_crs, osmnx_obj, name
    )


def _normalize_raster_data(data, dtype):
    """Normalize raster data to 0-255 range based on bit depth."""
    
    # Handle different data types
    if 'uint8' in dtype:
        return data.astype(np.float32)
    elif 'uint16' in dtype:
        return (data.astype(np.float32) / 256).clip(0, 255)
    elif 'int16' in dtype:
        # Often used for DEMs, may have negative values
        data_f = data.astype(np.float32)
        data_min = np.nanmin(data_f)
        data_max = np.nanmax(data_f)
        if data_max > data_min:
            return ((data_f - data_min) / (data_max - data_min) * 255).clip(0, 255)
        return np.zeros_like(data_f)
    elif 'float' in dtype or 'float32' in dtype or 'float64' in dtype:
        data_f = data.astype(np.float32)
        data_min = np.nanmin(data_f)
        data_max = np.nanmax(data_f)
        if data_max > data_min:
            return ((data_f - data_min) / (data_max - data_min) * 255).clip(0, 255)
        return np.zeros_like(data_f)
    else:
        # Default: assume 8-bit
        return data.astype(np.float32).clip(0, 255)


def _reproject_bounds(bounds, from_crs, to_crs):
    """Reproject bounds from one CRS to another."""
    if from_crs == to_crs:
        return bounds
    
    try:
        from pyproj import Transformer
        
        transformer = Transformer.from_crs(from_crs, to_crs, always_xy=True)
        
        # Transform corners
        west, south = transformer.transform(bounds['west'], bounds['south'])
        east, north = transformer.transform(bounds['east'], bounds['north'])
        
        return {
            'west': west,
            'south': south,
            'east': east,
            'north': north,
        }
    except ImportError:
        log("pyproj not installed, cannot reproject")
        return None
    except Exception as e:
        log(f"Error reprojecting: {e}")
        return None


def _create_textured_terrain_plane(image_path, bounds, source_crs, target_crs, 
                                   osmnx_obj, name):
    """
    Create a textured plane mesh for terrain visualization.
    
    Args:
        image_path: Path to texture image
        bounds: Dict with north, south, east, west
        source_crs: CRS of the bounds
        target_crs: Target CRS for positioning
        osmnx_obj: Optional OSMnx object to align with
        name: Object name
    
    Returns:
        Tuple (blender_object, metadata)
    """
    # Calculate dimensions
    lat_mid = (bounds['north'] + bounds['south']) / 2
    
    # For geographic CRS, convert to approximate meters
    if SUPPORTED_CRS.get(source_crs, {}).get('is_geographic', True):
        lat_km = 111.0
        lon_km = 111.0 * np.cos(np.radians(lat_mid))
        width_m = (bounds['east'] - bounds['west']) * lon_km * 1000
        height_m = (bounds['north'] - bounds['south']) * lat_km * 1000
    else:
        # Already in meters
        width_m = bounds['east'] - bounds['west']
        height_m = bounds['north'] - bounds['south']
    
    # Get scale from OSMnx object if available
    if osmnx_obj is not None:
        scale = osmnx_obj.get("osmnx_scale", 0.001)
    else:
        scale = 0.001
    
    width_bu = width_m * scale
    height_bu = height_m * scale
    
    log(f"Terrain plane: {width_m:.0f}m x {height_m:.0f}m -> {width_bu:.2f} x {height_bu:.2f} BU")
    
    # Create plane mesh
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    
    # Simple quad with correct aspect ratio
    half_w = width_bu / 2
    half_h = height_bu / 2
    
    verts = [
        (-half_w, -half_h, 0),
        (half_w, -half_h, 0),
        (half_w, half_h, 0),
        (-half_w, half_h, 0),
    ]
    faces = [(0, 1, 2, 3)]
    
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    # Add UV coordinates
    mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data
    
    # UV coordinates for the quad
    uv_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for i, uv in enumerate(uv_coords):
        uv_layer[i].uv = uv
    
    # Create object
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    # Load texture and create material
    mat = _create_terrain_texture_material(image_path, name)
    if mat:
        obj.data.materials.append(mat)
    
    # Position relative to OSMnx object
    if osmnx_obj is not None:
        obj.location = osmnx_obj.location.copy()
        obj.location.z -= 0.01  # Slightly below
        
        # Store alignment info
        obj["osmnx_parent"] = osmnx_obj.name
        osmnx_obj["terrain_plane_child"] = obj.name
    
    # Store metadata
    obj["is_terrain_plane"] = True
    obj["terrain_bounds_north"] = bounds['north']
    obj["terrain_bounds_south"] = bounds['south']
    obj["terrain_bounds_east"] = bounds['east']
    obj["terrain_bounds_west"] = bounds['west']
    obj["terrain_crs"] = source_crs
    obj["terrain_width_m"] = width_m
    obj["terrain_height_m"] = height_m
    
    metadata = {
        'bounds': bounds,
        'width_m': width_m,
        'height_m': height_m,
        'crs': source_crs,
    }
    
    log(f"Terrain plane created: {name}")
    
    return obj, metadata


def _create_terrain_texture_material(image_path, name):
    """Create a material with the terrain texture."""
    import os
    
    if not os.path.exists(image_path):
        log(f"Image not found: {image_path}")
        return None
    
    mat_name = f"{name}_Material"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Output
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    # Principled BSDF
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (100, 0)
    bsdf.inputs['Roughness'].default_value = 1.0
    bsdf.inputs['Specular IOR Level'].default_value = 0.0
    
    # Image texture
    tex_node = nodes.new(type='ShaderNodeTexImage')
    tex_node.location = (-300, 0)
    
    # Load image
    try:
        img = bpy.data.images.load(image_path)
        tex_node.image = img
        tex_node.interpolation = 'Linear'
    except Exception as e:
        log(f"Error loading texture: {e}")
        return None
    
    # UV Map
    uv_node = nodes.new(type='ShaderNodeUVMap')
    uv_node.location = (-500, 0)
    uv_node.uv_map = "UVMap"
    
    # Links
    links.new(uv_node.outputs['UV'], tex_node.inputs['Vector'])
    links.new(tex_node.outputs['Color'], bsdf.inputs['Base Color'])
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    return mat


def update_terrain_plane_offset(terrain_obj, offset_x=0, offset_y=0, offset_z=0, scale_xy=1.0):
    """
    Update the position and scale of a terrain plane.
    
    Args:
        terrain_obj: Terrain plane Blender object
        offset_x: X offset in Blender units
        offset_y: Y offset in Blender units
        offset_z: Z offset in Blender units
        scale_xy: Horizontal scale factor
    """
    if terrain_obj is None or not terrain_obj.get("is_terrain_plane", False):
        return
    
    # Get parent OSMnx object if exists
    parent_name = terrain_obj.get("osmnx_parent", "")
    if parent_name and parent_name in bpy.data.objects:
        parent = bpy.data.objects[parent_name]
        base_location = parent.location.copy()
    else:
        base_location = terrain_obj.location.copy()
    
    # Apply offset
    terrain_obj.location.x = base_location.x + offset_x
    terrain_obj.location.y = base_location.y + offset_y
    terrain_obj.location.z = base_location.z + offset_z - 0.01
    
    # Apply scale
    terrain_obj.scale.x = scale_xy
    terrain_obj.scale.y = scale_xy


def update_terrain_plane_opacity(terrain_obj, opacity):
    """Update terrain plane texture opacity."""
    if terrain_obj is None or not terrain_obj.get("is_terrain_plane", False):
        return
    
    if not terrain_obj.data.materials:
        return
    
    mat = terrain_obj.data.materials[0]
    if not mat.use_nodes:
        return
    
    # Find BSDF node and adjust alpha
    for node in mat.node_tree.nodes:
        if node.type == 'BSDF_PRINCIPLED':
            node.inputs['Alpha'].default_value = opacity
            mat.blend_method = 'BLEND' if opacity < 1.0 else 'OPAQUE'
            break

