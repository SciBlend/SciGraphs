import numpy as np
import pandas as pd
import bpy
import bmesh
from typing import Tuple, List, Dict, Optional
import os
import json

# Cache file for geocoded coordinates
GEOCODE_CACHE_FILE = os.path.join(
    os.path.dirname(__file__), 
    "..", 
    "..", 
    ".geocode_cache.json"
)

def load_geocode_cache() -> Dict[str, Tuple[float, float]]:
    """Load cached geocoded coordinates from file."""
    if os.path.exists(GEOCODE_CACHE_FILE):
        try:
            with open(GEOCODE_CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load geocode cache: {e}")
    return {}

def save_geocode_cache(cache: Dict[str, Tuple[float, float]]):
    """Save geocoded coordinates to cache file."""
    try:
        with open(GEOCODE_CACHE_FILE, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save geocode cache: {e}")

def detect_geospatial_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    """
    Auto-detect latitude and longitude columns by name patterns.
    Returns (lat_column, lon_column) or (None, None) if not found.
    """
    lat_patterns = ['lat', 'latitude', 'y', 'coord_y']
    lon_patterns = ['lon', 'long', 'longitude', 'x', 'coord_x']
    
    lat_col = None
    lon_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        
        if any(pattern in col_lower for pattern in lat_patterns) and lat_col is None:
            if pd.api.types.is_numeric_dtype(df[col]):
                lat_col = col
        
        if any(pattern in col_lower for pattern in lon_patterns) and lon_col is None:
            if pd.api.types.is_numeric_dtype(df[col]):
                lon_col = col
    
    if lat_col and lon_col:
        print(f"Detected geospatial columns: {lat_col}, {lon_col}")
        return lat_col, lon_col
    
    return None, None

def detect_country_columns(df: pd.DataFrame) -> bool:
    """
    Detect if dataframe contains country name columns.
    Returns True if likely country data is found.
    """
    country_patterns = ['country', 'nation', 'territory', 'origin', 'destination']
    
    # Known country names for validation
    known_countries = [
        'afghanistan', 'albania', 'algeria', 'australia', 'austria',
        'brazil', 'canada', 'china', 'france', 'germany', 'india',
        'italy', 'japan', 'mexico', 'russia', 'spain', 'turkey',
        'united kingdom', 'united states', 'usa', 'uk'
    ]
    
    for col in df.columns:
        col_lower = col.lower()
        
        if any(pattern in col_lower for pattern in country_patterns):
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                # Check if values look like country names
                sample_values = df[col].dropna().str.lower().unique()[:20]
                matches = sum(1 for val in sample_values if any(country in str(val) for country in known_countries))
                
                if matches >= 2:
                    print(f"Detected country column: {col}")
                    return True
    
    return False

def detect_temporal_columns(df: pd.DataFrame) -> Optional[str]:
    """
    Detect temporal/date columns.
    Returns the first temporal column found, or None.
    """
    time_patterns = ['year', 'date', 'time', 'month', 'day', 'period']
    
    for col in df.columns:
        col_lower = col.lower()
        
        if any(pattern in col_lower for pattern in time_patterns):
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
                print(f"Detected temporal column: {col}")
                return col
    
    return None

def geocode_locations(location_names: List[str], use_cache: bool = True) -> Dict[str, Tuple[float, float]]:
    """
    Convert location names to lat/lon coordinates using geopy.
    Uses caching to avoid repeated API calls.
    Returns dict mapping location_name -> (lat, lon).
    """
    try:
        from geopy.geocoders import Nominatim
        from geopy.exc import GeocoderTimedOut, GeocoderServiceError
        import time
    except ImportError:
        print("Error: geopy not available. Cannot geocode locations.")
        return {}
    
    # Load cache
    cache = load_geocode_cache() if use_cache else {}
    
    # Initialize geocoder
    geolocator = Nominatim(user_agent="scigraphs_blender_addon")
    
    results = {}
    unique_locations = list(set(location_names))
    
    print(f"Geocoding {len(unique_locations)} unique locations...")
    
    for i, location in enumerate(unique_locations):
        location_str = str(location).strip()
        
        if not location_str or location_str.lower() in ['nan', 'none', '']:
            continue
        
        # Check cache first
        if location_str in cache:
            results[location_str] = tuple(cache[location_str])
            continue
        
        # Geocode with retry logic
        retries = 3
        for attempt in range(retries):
            try:
                loc_result = geolocator.geocode(location_str, timeout=10)
                
                if loc_result:
                    lat, lon = loc_result.latitude, loc_result.longitude
                    results[location_str] = (lat, lon)
                    cache[location_str] = (lat, lon)
                    print(f"  [{i+1}/{len(unique_locations)}] {location_str}: ({lat:.2f}, {lon:.2f})")
                else:
                    print(f"  [{i+1}/{len(unique_locations)}] {location_str}: Not found")
                
                time.sleep(1.1)  # Respect rate limit
                break
                
            except (GeocoderTimedOut, GeocoderServiceError) as e:
                if attempt < retries - 1:
                    print(f"  Retry {attempt+1} for {location_str}...")
                    time.sleep(2)
                else:
                    print(f"  Failed to geocode {location_str}: {e}")
    
    # Save cache
    if use_cache and cache:
        save_geocode_cache(cache)
    
    print(f"Geocoded {len(results)}/{len(unique_locations)} locations")
    return results

def calculate_sphere_positions(
    lat_lon_dict: Dict[str, Tuple[float, float]], 
    radius: float = 5.0
) -> Dict[str, np.ndarray]:
    """
    Convert lat/lon coordinates to 3D positions on a sphere.
    
    Args:
        lat_lon_dict: Dictionary mapping node_name -> (latitude, longitude)
        radius: Radius of the sphere
    
    Returns:
        Dictionary mapping node_name -> [x, y, z] position
    """
    positions = {}
    
    for node_name, (lat, lon) in lat_lon_dict.items():
        # Convert degrees to radians
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        
        # Convert to Cartesian coordinates (Z-up)
        x = radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = radius * np.sin(lat_rad)
        
        positions[node_name] = np.array([x, y, z])
    
    return positions

def generate_great_circle_points(
    pos1: np.ndarray, 
    pos2: np.ndarray, 
    num_segments: int = 20
) -> np.ndarray:
    """
    Generate points along a great circle arc between two 3D positions on a sphere.
    
    Args:
        pos1: Starting position [x, y, z]
        pos2: Ending position [x, y, z]
        num_segments: Number of segments in the arc
    
    Returns:
        Array of shape (num_segments+1, 3) with arc points
    """
    # Normalize positions to unit sphere
    p1 = pos1 / np.linalg.norm(pos1)
    p2 = pos2 / np.linalg.norm(pos2)
    
    # Calculate angle between positions
    dot_product = np.clip(np.dot(p1, p2), -1.0, 1.0)
    angle = np.arccos(dot_product)
    
    # If points are very close, just return a straight line
    if angle < 0.01:
        t = np.linspace(0, 1, num_segments + 1)[:, np.newaxis]
        return pos1 + t * (pos2 - pos1)
    
    # Use spherical linear interpolation (slerp)
    t_values = np.linspace(0, 1, num_segments + 1)
    points = []
    
    radius = np.linalg.norm(pos1)
    
    for t in t_values:
        # Slerp formula
        sin_angle = np.sin(angle)
        a = np.sin((1 - t) * angle) / sin_angle
        b = np.sin(t * angle) / sin_angle
        
        point = (a * p1 + b * p2) * radius
        points.append(point)
    
    return np.array(points)

def filter_temporal_data(
    df: pd.DataFrame,
    time_col: str,
    aggregation: str = 'ALL',
    start: Optional[str] = None,
    end: Optional[str] = None,
    weight_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Filter and aggregate temporal data based on time column.
    
    Args:
        df: Input dataframe
        time_col: Name of time column
        aggregation: 'ALL', 'YEAR', 'MONTH', or 'RANGE'
        start: Start period for RANGE mode
        end: End period for RANGE mode
        weight_col: Column to aggregate (sum)
    
    Returns:
        Filtered/aggregated dataframe
    """
    df_copy = df.copy()
    
    # Handle different time column types
    if pd.api.types.is_numeric_dtype(df_copy[time_col]):
        # Assume it's a year
        df_copy['_year'] = df_copy[time_col].astype(int)
        df_copy['_period'] = df_copy['_year'].astype(str)
    else:
        # Try to parse as datetime
        try:
            df_copy[time_col] = pd.to_datetime(df_copy[time_col])
            df_copy['_year'] = df_copy[time_col].dt.year
            df_copy['_month'] = df_copy[time_col].dt.month
            df_copy['_period'] = df_copy['_year'].astype(str) + '-' + df_copy['_month'].astype(str).str.zfill(2)
        except Exception:
            print(f"Warning: Could not parse time column {time_col}")
            return df_copy
    
    # Filter by range if specified
    if aggregation == 'RANGE' and start and end:
        if '-' in start:
            # Month-level filtering
            df_copy = df_copy[
                (df_copy['_period'] >= start) & 
                (df_copy['_period'] <= end)
            ]
        else:
            # Year-level filtering
            start_year = int(start)
            end_year = int(end)
            df_copy = df_copy[
                (df_copy['_year'] >= start_year) & 
                (df_copy['_year'] <= end_year)
            ]
    
    # Aggregate based on mode
    if aggregation == 'ALL':
        # Aggregate all time periods
        group_cols = [col for col in df_copy.columns if col not in [time_col, weight_col, '_year', '_month', '_period']]
        
        if weight_col:
            df_result = df_copy.groupby(group_cols, as_index=False)[weight_col].sum()
        else:
            df_result = df_copy.drop_duplicates(subset=group_cols)
        
        return df_result
    
    elif aggregation == 'YEAR':
        # Keep yearly granularity
        group_cols = [col for col in df_copy.columns if col not in [time_col, weight_col, '_month', '_period']]
        
        if weight_col:
            df_result = df_copy.groupby(group_cols, as_index=False)[weight_col].sum()
        else:
            df_result = df_copy.drop_duplicates(subset=group_cols)
        
        return df_result
    
    else:
        # Keep original granularity (MONTH or RANGE with no aggregation)
        return df_copy

def create_globe_mesh(
    radius: float = 5.0,
    subdivisions: int = 64,
    material_style: str = 'SIMPLE',
    map_resolution: str = '110m',
    feature_type: str = 'COASTLINE',
    globe_theme: str = 'NONE',
    texture_resolution: str = '4K',
    water_specular: float = 0.8,
    water_roughness: float = 0.2,
    land_roughness: float = 0.8,
    bump_strength: float = 0.1
) -> bpy.types.Object:
    """
    Create a UV sphere mesh to represent Earth with various material styles.
    
    Creates a sphere with proper equirectangular UV mapping for satellite textures.
    The UV coordinates are aligned so that:
    - U (0 to 1) maps to longitude (-180 to +180 degrees)
    - V (0 to 1) maps to latitude (-90 to +90 degrees)
    
    Args:
        radius: Radius of the sphere
        subdivisions: Number of UV sphere segments (16-512, higher = smoother)
        material_style: Material style ('SIMPLE', 'OCEAN', 'WIREFRAME', 'TOPOGRAPHIC', 'WORLD_MAP')
        map_resolution: Map detail level for WORLD_MAP ('110m', '50m', '10m')
        feature_type: Type of geographic features ('LAND', 'COASTLINE', etc.)
        globe_theme: Texture theme for API-based textures ('NONE', 'NASA_BLUE_MARBLE', etc.)
        texture_resolution: Resolution for downloaded textures ('2K', '4K', '8K')
        water_specular: Specular intensity for ocean areas
        water_roughness: Surface roughness for ocean areas
        land_roughness: Surface roughness for land areas
        bump_strength: Intensity of bump/displacement mapping
    
    Returns:
        Created globe object
    """
    ring_count = max(16, subdivisions // 2)
    
    bpy.ops.mesh.primitive_uv_sphere_add(
        segments=subdivisions,
        ring_count=ring_count,
        radius=radius,
        location=(0, 0, 0)
    )
    
    globe_obj = bpy.context.active_object
    globe_obj.name = "Earth_Globe"
    
    print(f"  Created globe: {subdivisions} segments x {ring_count} rings = {len(globe_obj.data.vertices)} vertices")
    
    _ensure_equirectangular_uv(globe_obj, radius)
    
    if material_style == 'WORLD_MAP':
        print(f"  Computing geographic data (resolution: {map_resolution}, type: {feature_type})...")
        _compute_land_ocean_attribute(globe_obj, radius, map_resolution, feature_type)
    
    if globe_theme != 'NONE':
        mat = _create_textured_pbr_material(
            globe_theme,
            texture_resolution,
            water_specular,
            water_roughness,
            land_roughness,
            bump_strength,
            globe_obj
        )
    else:
        mat = _create_globe_material(material_style, radius)
    
    if globe_obj.data.materials:
        globe_obj.data.materials[0] = mat
    else:
        globe_obj.data.materials.append(mat)
    
    return globe_obj


def _ensure_equirectangular_uv(globe_obj: bpy.types.Object, radius: float):
    """
    Ensure the globe mesh has proper equirectangular UV mapping.
    
    This function recalculates UVs to ensure:
    - U coordinate (0-1) maps exactly to longitude (-180 to +180 degrees)
    - V coordinate (0-1) maps exactly to latitude (-90 to +90 degrees)
    
    This is critical for satellite textures to align correctly with geographic data.
    
    Args:
        globe_obj: The globe mesh object
        radius: Radius of the globe (used for coordinate conversion)
    """
    import math
    
    mesh = globe_obj.data
    
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="UVMap")
    
    uv_layer = mesh.uv_layers.active.data
    
    for poly in mesh.polygons:
        for loop_idx in poly.loop_indices:
            vert_idx = mesh.loops[loop_idx].vertex_index
            vert = mesh.vertices[vert_idx]
            
            x, y, z = vert.co
            
            r = math.sqrt(x*x + y*y + z*z)
            if r < 0.0001:
                lat = 0.0
                lon = 0.0
            else:
                lat = math.asin(max(-1.0, min(1.0, z / r)))
                lon = math.atan2(y, x)
            
            u = (lon + math.pi) / (2.0 * math.pi)
            v = (lat + math.pi / 2.0) / math.pi
            
            uv_layer[loop_idx].uv = (u, v)
    
    _fix_uv_seam(mesh)


def _fix_uv_seam(mesh):
    """
    Fix the UV seam at the 180/-180 degree longitude line.
    
    Blender UV spheres can have issues at the seam where U wraps from 1 to 0.
    This function detects faces crossing the seam and adjusts their UVs.
    
    Args:
        mesh: The mesh data to fix
    """
    uv_layer = mesh.uv_layers.active.data
    
    for poly in mesh.polygons:
        loop_indices = list(poly.loop_indices)
        us = [uv_layer[li].uv[0] for li in loop_indices]
        
        if max(us) - min(us) > 0.5:
            for li in loop_indices:
                u, v = uv_layer[li].uv
                if u < 0.5:
                    uv_layer[li].uv = (u + 1.0, v)


def _create_globe_material(material_style: str, globe_radius: float):
    """
    Create a material for the Earth globe based on the selected style.
    
    Args:
        material_style: Material style identifier
        globe_radius: Radius of the globe (used for scaling textures)
    
    Returns:
        Created material
    """
    mat = bpy.data.materials.new(name=f"Globe_{material_style}")
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Output node (always needed)
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (600, 0)
    
    # Create different materials based on style
    if material_style == 'SIMPLE':
        _create_simple_material(nodes, links, output_node)
        mat.blend_method = 'BLEND'
    
    elif material_style == 'OCEAN':
        _create_ocean_material(nodes, links, output_node)
        mat.blend_method = 'BLEND'
    
    elif material_style == 'WIREFRAME':
        _create_wireframe_material(nodes, links, output_node)
        mat.blend_method = 'BLEND'
    
    elif material_style == 'TOPOGRAPHIC':
        _create_topographic_material(nodes, links, output_node)
        mat.blend_method = 'BLEND'
    
    elif material_style == 'WORLD_MAP':
        _create_world_map_material(nodes, links, output_node)
        mat.blend_method = 'BLEND'
    
    # Shadow method changed in Blender 4.5+
    if hasattr(mat, 'shadow_method'):
        mat.shadow_method = 'NONE'
    
    return mat


def _create_simple_material(nodes, links, output_node):
    """Create simple blue ocean material."""
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (300, 0)
    
    principled.inputs['Base Color'].default_value = (0.1, 0.2, 0.4, 1.0)
    principled.inputs['Metallic'].default_value = 0.0
    principled.inputs['Roughness'].default_value = 0.4
    principled.inputs['Alpha'].default_value = 0.3
    
    links.new(principled.outputs['BSDF'], output_node.inputs['Surface'])


def _create_ocean_material(nodes, links, output_node):
    """Create ocean material with procedural waves."""
    # Texture coordinate
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-600, 0)
    
    # Wave texture
    wave = nodes.new(type='ShaderNodeTexWave')
    wave.location = (-400, 0)
    wave.wave_type = 'RINGS'
    wave.inputs['Scale'].default_value = 20.0
    wave.inputs['Distortion'].default_value = 2.0
    wave.inputs['Detail'].default_value = 4.0
    
    # Color ramp for wave colors
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    color_ramp.location = (-200, 0)
    color_ramp.color_ramp.elements[0].color = (0.05, 0.15, 0.35, 1.0)  # Deep ocean
    color_ramp.color_ramp.elements[1].color = (0.15, 0.3, 0.5, 1.0)    # Shallow ocean
    
    # Principled BSDF
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (300, 0)
    principled.inputs['Metallic'].default_value = 0.1
    principled.inputs['Roughness'].default_value = 0.3
    principled.inputs['Alpha'].default_value = 0.4
    
    # Link nodes
    links.new(tex_coord.outputs['Object'], wave.inputs['Vector'])
    links.new(wave.outputs['Color'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], principled.inputs['Base Color'])
    links.new(principled.outputs['BSDF'], output_node.inputs['Surface'])


def _create_wireframe_material(nodes, links, output_node):
    """Create wireframe material showing grid lines."""
    # Wireframe node
    wireframe = nodes.new(type='ShaderNodeWireframe')
    wireframe.location = (-200, 0)
    wireframe.inputs['Size'].default_value = 0.01
    
    # Mix shader for wireframe
    mix = nodes.new(type='ShaderNodeMixShader')
    mix.location = (300, 0)
    
    # Transparent shader for interior
    transparent = nodes.new(type='ShaderNodeBsdfTransparent')
    transparent.location = (100, 100)
    
    # Emission shader for wireframe lines
    emission = nodes.new(type='ShaderNodeEmission')
    emission.location = (100, -100)
    emission.inputs['Color'].default_value = (0.2, 0.4, 0.8, 1.0)
    emission.inputs['Strength'].default_value = 1.0
    
    # Link nodes
    links.new(wireframe.outputs['Fac'], mix.inputs['Fac'])
    links.new(transparent.outputs['BSDF'], mix.inputs[1])
    links.new(emission.outputs['Emission'], mix.inputs[2])
    links.new(mix.outputs['Shader'], output_node.inputs['Surface'])


def _create_topographic_material(nodes, links, output_node):
    """Create topographic material with height-based colors."""
    # Texture coordinate
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-800, 0)
    
    # Noise texture for terrain
    noise = nodes.new(type='ShaderNodeTexNoise')
    noise.location = (-600, 0)
    noise.inputs['Scale'].default_value = 5.0
    noise.inputs['Detail'].default_value = 8.0
    noise.inputs['Roughness'].default_value = 0.5
    
    # Color ramp for elevation colors
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    color_ramp.location = (-400, 0)
    
    # Add multiple color stops for terrain
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0.0, 0.1, 0.3, 1.0)  # Deep ocean
    
    color_ramp.color_ramp.elements.new(0.3)
    color_ramp.color_ramp.elements[1].color = (0.1, 0.3, 0.5, 1.0)  # Shallow ocean
    
    color_ramp.color_ramp.elements.new(0.5)
    color_ramp.color_ramp.elements[2].color = (0.9, 0.85, 0.7, 1.0)  # Beach
    
    color_ramp.color_ramp.elements.new(0.7)
    color_ramp.color_ramp.elements[3].color = (0.2, 0.5, 0.1, 1.0)  # Lowlands
    
    color_ramp.color_ramp.elements[4].position = 1.0
    color_ramp.color_ramp.elements[4].color = (0.9, 0.9, 0.9, 1.0)  # Mountains
    
    # Principled BSDF
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (300, 0)
    principled.inputs['Roughness'].default_value = 0.6
    principled.inputs['Alpha'].default_value = 0.5
    
    # Link nodes
    links.new(tex_coord.outputs['Object'], noise.inputs['Vector'])
    links.new(noise.outputs['Fac'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], principled.inputs['Base Color'])
    links.new(principled.outputs['BSDF'], output_node.inputs['Surface'])


def _create_world_map_material(nodes, links, output_node):
    """
    Create material using vertex attribute to distinguish land from ocean.
    Note: This requires the globe mesh to have 'is_land' attribute pre-computed.
    """
    attr_node = nodes.new(type='ShaderNodeAttribute')
    attr_node.location = (-400, 0)
    attr_node.attribute_name = 'is_land'
    
    color_ramp = nodes.new(type='ShaderNodeValToRGB')
    color_ramp.location = (-200, 0)
    
    color_ramp.color_ramp.elements[0].position = 0.0
    color_ramp.color_ramp.elements[0].color = (0.05, 0.15, 0.4, 1.0)
    
    color_ramp.color_ramp.elements[1].position = 1.0
    color_ramp.color_ramp.elements[1].color = (0.4, 0.6, 0.3, 1.0)
    
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (300, 0)
    principled.inputs['Roughness'].default_value = 0.6
    principled.inputs['Alpha'].default_value = 0.4
    
    links.new(attr_node.outputs['Fac'], color_ramp.inputs['Fac'])
    links.new(color_ramp.outputs['Color'], principled.inputs['Base Color'])
    links.new(principled.outputs['BSDF'], output_node.inputs['Surface'])


def _create_textured_pbr_material(
    globe_theme: str,
    texture_resolution: str,
    water_specular: float,
    water_roughness: float,
    land_roughness: float,
    bump_strength: float,
    globe_obj: bpy.types.Object
) -> bpy.types.Material:
    """
    Create a PBR material with satellite/map texture for realistic globe rendering.
    
    This function downloads the appropriate texture (if not cached), creates a
    Principled BSDF material, and configures it with:
    - Base color from satellite imagery
    - Different roughness for land vs ocean (using is_land attribute if available)
    - Optional bump mapping for surface relief
    - Emission for night lights (NASA VIIRS theme)
    
    Args:
        globe_theme: Texture theme identifier
        texture_resolution: Resolution of texture to download
        water_specular: Specular reflection for ocean
        water_roughness: Surface roughness for ocean
        land_roughness: Surface roughness for land
        bump_strength: Intensity of bump mapping
        globe_obj: The globe mesh object (for checking attributes)
    
    Returns:
        Configured Blender material
    """
    from . import texture_api
    
    texture_path, material_hints = texture_api.get_texture_for_globe(
        globe_theme, texture_resolution
    )
    
    mat = bpy.data.materials.new(name=f"Globe_{globe_theme}")
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    output_node = nodes.new(type='ShaderNodeOutputMaterial')
    output_node.location = (800, 0)
    
    principled = nodes.new(type='ShaderNodeBsdfPrincipled')
    principled.location = (500, 0)
    
    tex_coord = nodes.new(type='ShaderNodeTexCoord')
    tex_coord.location = (-600, 200)
    
    if texture_path:
        image_node = nodes.new(type='ShaderNodeTexImage')
        image_node.location = (-200, 200)
        
        try:
            img = bpy.data.images.load(texture_path)
            image_node.image = img
            image_node.interpolation = 'Smart'
            print(f"  Loaded texture: {texture_path}")
        except Exception as e:
            print(f"  Warning: Could not load texture: {e}")
            image_node = None
    else:
        image_node = None
    
    if image_node:
        links.new(tex_coord.outputs['UV'], image_node.inputs['Vector'])
        links.new(image_node.outputs['Color'], principled.inputs['Base Color'])
        
        if globe_theme == 'NASA_VIIRS':
            emission_strength = material_hints.get('emission_strength', 0.8)
            principled.inputs['Emission Strength'].default_value = emission_strength
            links.new(image_node.outputs['Color'], principled.inputs['Emission Color'])
    else:
        principled.inputs['Base Color'].default_value = (0.1, 0.3, 0.6, 1.0)
    
    has_land_attr = 'is_land' in [attr.name for attr in globe_obj.data.attributes]
    
    if has_land_attr:
        attr_node = nodes.new(type='ShaderNodeAttribute')
        attr_node.location = (-400, -200)
        attr_node.attribute_name = 'is_land'
        
        roughness_ramp = nodes.new(type='ShaderNodeValToRGB')
        roughness_ramp.location = (-100, -200)
        roughness_ramp.color_ramp.elements[0].position = 0.0
        roughness_ramp.color_ramp.elements[0].color = (water_roughness, water_roughness, water_roughness, 1.0)
        roughness_ramp.color_ramp.elements[1].position = 1.0
        roughness_ramp.color_ramp.elements[1].color = (land_roughness, land_roughness, land_roughness, 1.0)
        
        links.new(attr_node.outputs['Fac'], roughness_ramp.inputs['Fac'])
        links.new(roughness_ramp.outputs['Color'], principled.inputs['Roughness'])
        
        specular_ramp = nodes.new(type='ShaderNodeValToRGB')
        specular_ramp.location = (-100, -400)
        specular_ramp.color_ramp.elements[0].position = 0.0
        specular_ramp.color_ramp.elements[0].color = (water_specular, water_specular, water_specular, 1.0)
        specular_ramp.color_ramp.elements[1].position = 1.0
        specular_ramp.color_ramp.elements[1].color = (0.2, 0.2, 0.2, 1.0)
        
        links.new(attr_node.outputs['Fac'], specular_ramp.inputs['Fac'])
        
        if 'Specular IOR Level' in principled.inputs:
            links.new(specular_ramp.outputs['Color'], principled.inputs['Specular IOR Level'])
    else:
        avg_roughness = (water_roughness + land_roughness) / 2.0
        principled.inputs['Roughness'].default_value = avg_roughness
    
    if bump_strength > 0 and image_node:
        bump_node = nodes.new(type='ShaderNodeBump')
        bump_node.location = (200, -300)
        bump_node.inputs['Strength'].default_value = bump_strength
        
        rgb_to_bw = nodes.new(type='ShaderNodeRGBToBW')
        rgb_to_bw.location = (0, -300)
        
        links.new(image_node.outputs['Color'], rgb_to_bw.inputs['Color'])
        links.new(rgb_to_bw.outputs['Val'], bump_node.inputs['Height'])
        links.new(bump_node.outputs['Normal'], principled.inputs['Normal'])
    
    links.new(principled.outputs['BSDF'], output_node.inputs['Surface'])
    
    if globe_theme == 'DATA_OVERLAY':
        mat.blend_method = 'BLEND'
        principled.inputs['Alpha'].default_value = material_hints.get('alpha', 0.3)
    else:
        mat.blend_method = 'OPAQUE'
    
    if hasattr(mat, 'shadow_method'):
        mat.shadow_method = 'NONE'
    
    return mat


def _compute_land_ocean_attribute(globe_obj: bpy.types.Object, radius: float, map_resolution: str = '110m', feature_type: str = 'COASTLINE'):
    """
    Compute 'is_land' attribute for each vertex of the globe mesh.
    Uses geopandas to check if each vertex falls on land or ocean.
    
    Args:
        globe_obj: The globe mesh object
        radius: Radius of the globe
        map_resolution: Map detail level ('110m', '50m', '10m')
        feature_type: Type of features to load ('LAND', 'COASTLINE', 'LAND_OCEAN', 'BATHYMETRY', 'RIVERS_LAKES')
    """
    import geopandas as gpd
    from shapely.geometry import Point
    import math
    import os
    import urllib.request
    import zipfile
    
    # Map resolution to download size info
    resolution_info = {
        '110m': ('~1 MB', 'low'),
        '50m': ('~3 MB', 'medium'),
        '10m': ('~20 MB', 'high')
    }
    size_info, quality = resolution_info.get(map_resolution, ('~1 MB', 'low'))
    
    # Map feature types to Natural Earth dataset names and categories
    feature_datasets = {
        'LAND': ('physical', f'ne_{map_resolution}_land'),
        'COASTLINE': ('physical', f'ne_{map_resolution}_coastline'),
        'LAND_OCEAN': ('physical', f'ne_{map_resolution}_ocean'),
        'BATHYMETRY': ('physical', f'ne_{map_resolution}_bathymetry_all'),
        'RIVERS_LAKES': ('physical', f'ne_{map_resolution}_rivers_lake_centerlines'),
    }
    
    category, dataset_name = feature_datasets.get(feature_type, ('cultural', f'ne_{map_resolution}_admin_0_countries'))
    
    print(f"    Loading {feature_type} data ({quality} quality, {map_resolution} scale)...")
    
    # Build file paths based on resolution and feature type
    cache_dir = os.path.join(os.path.dirname(__file__), '..', '.naturalearth_cache')
    cache_file = os.path.join(cache_dir, f'{dataset_name}.shp')
    
    world = None
    
    # Check if cache exists for this resolution and feature type
    if os.path.exists(cache_file):
        try:
            print(f"    Loading from cache: {cache_file}")
            world = gpd.read_file(cache_file)
        except Exception as e:
            print(f"    Warning: Could not load cached data: {e}")
            print("    Will try to download...")
    
    # If cache doesn't exist or failed, download from Natural Earth
    if world is None:
        try:
            print(f"    Downloading Natural Earth {feature_type} data ({map_resolution}, {size_info})...")
            os.makedirs(cache_dir, exist_ok=True)
            
            # Natural Earth CDN URL
            url = f"https://naciscdn.org/naturalearth/{map_resolution}/{category}/{dataset_name}.zip"
            zip_path = os.path.join(cache_dir, f'{dataset_name}.zip')
            
            # Create request with headers to avoid 406 error
            req = urllib.request.Request(
                url,
                headers={
                    'User-Agent': 'SciGraphs-Blender-Addon/1.0 (Blender)',
                    'Accept': '*/*'
                }
            )
            
            # Download with proper headers
            print(f"    Downloading from: {url}")
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                out_file.write(response.read())
            
            print("    Download complete, extracting...")
            
            # Extract
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)
            
            # Remove zip file
            os.remove(zip_path)
            
            print(f"    {feature_type} {map_resolution} data cached for future use")
            world = gpd.read_file(cache_file)
            
        except Exception as e:
            print(f"    Error downloading Natural Earth data: {e}")
            print(f"    Trying fallback: using land polygons...")
            
            # Fallback to land polygons if requested dataset doesn't exist
            fallback_name = f'ne_{map_resolution}_land'
            fallback_file = os.path.join(cache_dir, f'{fallback_name}.shp')
            
            if os.path.exists(fallback_file):
                try:
                    world = gpd.read_file(fallback_file)
                    print(f"    Loaded fallback data successfully")
                except:
                    pass
            
            if world is None:
                print(f"    Using final fallback: all vertices as ocean")
                # Create attribute with all zeros (ocean)
                attr = globe_obj.data.attributes.new(name='is_land', type='FLOAT', domain='POINT')
                return
    
    # Create a single geometry combining all land masses for faster checking
    land_union = world.geometry.unary_union
    
    # Create the vertex attribute
    attr = globe_obj.data.attributes.new(name='is_land', type='FLOAT', domain='POINT')
    
    # Process each vertex
    mesh = globe_obj.data
    total_verts = len(mesh.vertices)
    print(f"    Checking {total_verts} vertices...")
    
    for vert_idx, vert in enumerate(mesh.vertices):
        # Get vertex position in 3D space
        x, y, z = vert.co
        
        # Convert 3D Cartesian to latitude/longitude
        # Assuming Z is up, X is 0° longitude, Y is 90° longitude
        lat = math.degrees(math.asin(z / radius))
        lon = math.degrees(math.atan2(y, x))
        
        # Create a point and check if it's in land
        point = Point(lon, lat)
        
        # Check if point is within land
        is_land = land_union.contains(point)
        
        # Set attribute value (1.0 for land, 0.0 for ocean)
        attr.data[vert_idx].value = 1.0 if is_land else 0.0
        
        # Progress reporting
        if (vert_idx + 1) % 500 == 0:
            print(f"      Processed {vert_idx + 1}/{total_verts} vertices...")
    
    print(f"    Land/ocean attribute computed for {total_verts} vertices")

