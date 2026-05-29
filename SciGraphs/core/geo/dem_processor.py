# DEM Processing module
#
# Handles elevation data processing, nodata interpolation,
# normalization, and mesh generation from raster data

import bpy
import bmesh
import numpy as np
from ...utils.logger import log


def fill_nodata(data, nodata_value=None, method='nearest'):
    """
    Fill NoData values in elevation array using interpolation.
    
    Args:
        data: 2D numpy array of elevation values
        nodata_value: Value representing NoData (None to auto-detect)
        method: Interpolation method ('nearest', 'linear')
    
    Returns:
        Array with NoData values filled
    """
    result = data.copy().astype(np.float32)
    
    # Create mask of invalid values
    if nodata_value is not None:
        invalid_mask = (result == nodata_value)
    else:
        invalid_mask = np.isnan(result) | np.isinf(result)
    
    invalid_count = np.sum(invalid_mask)
    if invalid_count == 0:
        return result
    
    log(f"Filling {invalid_count} NoData values...")
    
    # Mark invalid as NaN for processing
    result[invalid_mask] = np.nan
    
    valid_mask = ~np.isnan(result)
    if not np.any(valid_mask):
        log("Warning: No valid data found")
        return np.zeros_like(result)
    
    if method == 'nearest':
        # Use scipy distance transform for nearest neighbor
        try:
            from scipy import ndimage
            
            indices = ndimage.distance_transform_edt(
                np.isnan(result),
                return_distances=False,
                return_indices=True
            )
            result = result[indices[0], indices[1]]
            
        except ImportError:
            # Fallback: fill with mean
            mean_val = np.nanmean(result)
            result[np.isnan(result)] = mean_val
    
    elif method == 'linear':
        # Linear interpolation using griddata
        try:
            from scipy.interpolate import griddata
            
            height, width = result.shape
            y_coords, x_coords = np.mgrid[0:height, 0:width]
            
            valid_points = np.column_stack([
                y_coords[valid_mask].ravel(),
                x_coords[valid_mask].ravel()
            ])
            valid_values = result[valid_mask].ravel()
            
            invalid_points = np.column_stack([
                y_coords[~valid_mask].ravel(),
                x_coords[~valid_mask].ravel()
            ])
            
            if len(invalid_points) > 0:
                interpolated = griddata(
                    valid_points,
                    valid_values,
                    invalid_points,
                    method='linear',
                    fill_value=np.nanmean(result)
                )
                result[~valid_mask] = interpolated
                
        except ImportError:
            mean_val = np.nanmean(result)
            result[np.isnan(result)] = mean_val
    
    return result


def normalize_elevation(data, dtype, target_range=(0, 1)):
    """
    Normalize elevation data based on bit depth.
    
    Args:
        data: 2D numpy array
        dtype: Data type string ('uint8', 'int16', 'float32', etc.)
        target_range: Tuple of (min, max) for output range
    
    Returns:
        Normalized array as float32
    """
    data_f = data.astype(np.float32)
    
    # Get value range based on dtype
    if 'uint8' in dtype:
        data_min, data_max = 0, 255
    elif 'int16' in dtype:
        # For DEMs, use actual data range
        data_min = np.nanmin(data_f)
        data_max = np.nanmax(data_f)
    elif 'uint16' in dtype:
        data_min = np.nanmin(data_f)
        data_max = np.nanmax(data_f)
    elif 'float' in dtype:
        data_min = np.nanmin(data_f)
        data_max = np.nanmax(data_f)
    else:
        data_min = np.nanmin(data_f)
        data_max = np.nanmax(data_f)
    
    if data_max == data_min:
        return np.zeros_like(data_f)
    
    # Normalize to target range
    t_min, t_max = target_range
    normalized = (data_f - data_min) / (data_max - data_min)
    normalized = normalized * (t_max - t_min) + t_min
    
    return normalized


def calculate_displace_strength(georaster, scale=1.0):
    """
    Calculate appropriate displacement strength based on raster properties.
    
    The strength determines how much the modifier displaces vertices.
    For real-world elevation data, 1 unit of displacement should equal
    1 meter of elevation (adjusted by scale factor).
    
    Args:
        georaster: GeoRaster instance
        scale: Scale factor (e.g., 0.001 for km to Blender units)
    
    Returns:
        Tuple (strength, midlevel) for Displace modifier
    """
    stats = georaster.get_statistics()
    if stats is None:
        return 1.0, 0.5
    
    elev_min = stats['min']
    elev_max = stats['max']
    elev_range = elev_max - elev_min
    
    dtype = georaster.dtype
    
    # Calculate strength based on data type and range
    if 'float' in dtype:
        # Float data: values are typically in meters
        # Strength = range * scale
        strength = elev_range * scale
        midlevel = (elev_min - elev_min) / elev_range if elev_range > 0 else 0
        
    elif 'int16' in dtype:
        # Int16: common for SRTM data, values in meters
        strength = elev_range * scale
        midlevel = 0
        
    elif 'uint16' in dtype:
        # Uint16: might be scaled
        strength = elev_range * scale
        midlevel = 0
        
    elif 'uint8' in dtype:
        # 8-bit: usually normalized or scaled
        # Assume values represent meters if range is reasonable
        if elev_range > 10:
            strength = elev_range * scale
        else:
            # Likely normalized 0-1 or 0-255
            strength = 255 * scale
        midlevel = 0
        
    else:
        strength = elev_range * scale
        midlevel = 0
    
    log(f"Displace strength: {strength:.4f}, midlevel: {midlevel:.4f}")
    log(f"Elevation range: {elev_min:.1f} - {elev_max:.1f} m ({elev_range:.1f} m)")
    
    return strength, midlevel


def create_raster_extent_mesh(georaster, scale=0.001, name="DEM_Plane", osmnx_obj=None):
    """
    Create a simple plane mesh matching the raster extent.
    
    Args:
        georaster: GeoRaster instance
        scale: Scale factor for coordinates
        name: Name for the mesh
        osmnx_obj: Optional OSMnx object to align with
    
    Returns:
        Blender mesh object
    """
    bounds = georaster.bounds
    if bounds is None:
        log("GeoRaster has no bounds")
        return None
    
    # Calculate center - use OSMnx center if available
    raster_center_lon = (bounds['east'] + bounds['west']) / 2
    raster_center_lat = (bounds['north'] + bounds['south']) / 2
    
    graph_center_lon = raster_center_lon
    graph_center_lat = raster_center_lat
    
    if osmnx_obj is not None and osmnx_obj.get("is_osmnx", False):
        # Get stored center coordinates from OSMnx object
        stored_lat = osmnx_obj.get("osmnx_center_lat")
        stored_lon = osmnx_obj.get("osmnx_center_lon")
        
        if stored_lat is not None and stored_lon is not None:
            graph_center_lat = stored_lat
            graph_center_lon = stored_lon
    
    # Calculate dimensions
    if georaster.is_geographic_crs():
        lat_m = 111000  # meters per degree latitude
        lon_m = 111000 * np.cos(np.radians(graph_center_lat))
        
        width_m = (bounds['east'] - bounds['west']) * lon_m
        height_m = (bounds['north'] - bounds['south']) * lat_m
        
        # Offset from graph center to raster center
        offset_x_m = (raster_center_lon - graph_center_lon) * lon_m
        offset_y_m = (raster_center_lat - graph_center_lat) * lat_m
    else:
        width_m = bounds['east'] - bounds['west']
        height_m = bounds['north'] - bounds['south']
        offset_x_m = 0
        offset_y_m = 0
    
    # Scale to Blender units
    width = width_m * scale
    height = height_m * scale
    offset_x = offset_x_m * scale
    offset_y = offset_y_m * scale
    
    # Create mesh centered at offset position
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    
    half_w = width / 2
    half_h = height / 2
    
    verts = [
        (offset_x - half_w, offset_y - half_h, 0),
        (offset_x + half_w, offset_y - half_h, 0),
        (offset_x + half_w, offset_y + half_h, 0),
        (offset_x - half_w, offset_y + half_h, 0),
    ]
    faces = [(0, 1, 2, 3)]
    
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    # Add UV coordinates
    mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data
    uv_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for i, uv in enumerate(uv_coords):
        uv_layer[i].uv = uv
    
    # Create object
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    # Store metadata
    obj["is_dem_terrain"] = True
    obj["dem_width_m"] = width_m
    obj["dem_height_m"] = height_m
    obj["dem_scale"] = scale
    # Projection origin (lat/lon of the local equirectangular frame
    # used to position vertices). Required by the basemap UV unwrapper
    # to invert mesh-XY back to (lat, lon).
    obj["dem_center_lat"] = float(graph_center_lat)
    obj["dem_center_lon"] = float(graph_center_lon)
    obj["dem_bounds_north"] = bounds['north']
    obj["dem_bounds_south"] = bounds['south']
    obj["dem_bounds_east"] = bounds['east']
    obj["dem_bounds_west"] = bounds['west']
    
    return obj


def apply_displace_modifier(obj, georaster, subdivision_levels=6, scale=0.001):
    """
    Apply Subdivision Surface and Displace modifiers to create terrain.
    
    This is the "fast" method that uses Blender's modifier system.
    
    Args:
        obj: Blender mesh object (plane)
        georaster: GeoRaster instance
        subdivision_levels: Number of subdivision levels
        scale: Scale factor for displacement
    
    Returns:
        True on success
    """
    if obj is None or georaster is None:
        return False
    
    # Load image as texture
    try:
        img = bpy.data.images.load(georaster.filepath)
        img.colorspace_settings.name = 'Non-Color'
    except Exception as e:
        log(f"Error loading DEM as image: {e}")
        return False
    
    # Create texture
    tex_name = f"{obj.name}_DEM_Tex"
    tex = bpy.data.textures.new(tex_name, type='IMAGE')
    tex.image = img
    tex.extension = 'EXTEND'
    
    # Add Subdivision Surface modifier
    subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.subdivision_type = 'SIMPLE'
    subsurf.levels = subdivision_levels
    subsurf.render_levels = subdivision_levels
    
    # Add Displace modifier
    displace = obj.modifiers.new(name="DEM_Displace", type='DISPLACE')
    displace.texture = tex
    displace.texture_coords = 'UV'
    displace.direction = 'Z'
    
    # Calculate appropriate strength
    strength, midlevel = calculate_displace_strength(georaster, scale)
    displace.strength = strength
    displace.mid_level = midlevel
    
    # Store reference
    obj["dem_texture"] = tex_name
    obj["dem_image"] = img.name
    
    log(f"Applied Displace modifier with {subdivision_levels} subdivisions")
    
    return True


def raster_to_mesh(georaster, scale=0.001, subsample=1, name="DEM_RawMesh",
                   osmnx_obj=None, vertical_scale=1.0, vertical_offset=0.0):
    """
    Convert raster directly to mesh vertices (Raw Mesh method).
    
    This creates a vertex for each pixel, with Z = elevation.
    Slower but more accurate than the Displace method.
    
    Args:
        georaster: GeoRaster instance
        scale: Scale factor for coordinates
        subsample: Subsample factor (2 = half resolution, etc.)
        name: Name for the mesh
        osmnx_obj: Optional OSMnx object to align with (uses same center)
        vertical_scale: Vertical exaggeration factor
        vertical_offset: Base elevation offset in meters
    
    Returns:
        Blender mesh object
    """
    if georaster is None or georaster.data is None:
        return None
    
    data = georaster.data
    
    # Subsample if requested
    if subsample > 1:
        data = data[::subsample, ::subsample]
    
    height, width = data.shape
    log(f"Building raw mesh: {width}x{height} vertices")
    
    # Fill nodata values
    data = fill_nodata(data, georaster.nodata)
    
    # Get geotransform
    if georaster.geotransform:
        ox, sx, _, oy, _, sy = georaster.geotransform
        sx *= subsample
        sy *= subsample
    else:
        # Fallback: assume 1 unit per pixel
        ox, oy = 0, 0
        sx, sy = 1, -1
    
    # Calculate center - use OSMnx center if available for alignment
    if osmnx_obj is not None and osmnx_obj.get("is_osmnx", False):
        # Get stored center coordinates from OSMnx object
        center_y = osmnx_obj.get("osmnx_center_lat")
        center_x = osmnx_obj.get("osmnx_center_lon")
        
        if center_y is not None and center_x is not None:
            log(f"Using stored OSMnx center: {center_y:.5f}, {center_x:.5f}")
        else:
            # Fallback: calculate from raster bounds
            center_x = (georaster.bounds['east'] + georaster.bounds['west']) / 2
            center_y = (georaster.bounds['north'] + georaster.bounds['south']) / 2
    else:
        center_x = (georaster.bounds['east'] + georaster.bounds['west']) / 2
        center_y = (georaster.bounds['north'] + georaster.bounds['south']) / 2
    
    # Conversion factors for geographic CRS
    if georaster.is_geographic_crs():
        lat_m = 111000  # meters per degree latitude
        lon_m = 111000 * np.cos(np.radians(center_y))
    else:
        lat_m = 1
        lon_m = 1
    
    # Get elevation stats for reference
    elev_min = np.nanmin(data)
    elev_max = np.nanmax(data)
    
    # Build vertices
    verts = []
    for row in range(height):
        for col in range(width):
            # Geographic coordinates
            if georaster.is_geographic_crs():
                geo_x = ox + col * sx
                geo_y = oy + row * sy
                
                # Convert to meters relative to center
                x_m = (geo_x - center_x) * lon_m
                y_m = (geo_y - center_y) * lat_m
            else:
                x_m = ox + col * sx - center_x
                y_m = oy + row * sy - center_y
            
            # Elevation - use same formula as graph for alignment
            z_m = float(data[row, col])
            
            # Scale to Blender units
            # Use same formula as apply_georaster_elevations_to_graph
            x = x_m * scale
            y = y_m * scale
            z = ((z_m - elev_min) + vertical_offset) * scale * vertical_scale
            
            verts.append((x, y, z))
    
    # Build faces (quads)
    faces = []
    for row in range(height - 1):
        for col in range(width - 1):
            v0 = row * width + col
            v1 = v0 + 1
            v2 = v0 + width + 1
            v3 = v0 + width
            faces.append((v0, v1, v2, v3))
    
    # Create mesh
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()
    
    # Smooth shading
    for polygon in mesh.polygons:
        polygon.use_smooth = True
    
    # Create object
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    # Add elevation attribute
    if "elevation" in mesh.attributes:
        mesh.attributes.remove(mesh.attributes["elevation"])
    
    elev_attr = mesh.attributes.new(name="elevation", type='FLOAT', domain='POINT')
    elevations = data.flatten().astype(np.float32)
    elev_attr.data.foreach_set("value", elevations)
    
    # Store metadata
    obj["is_dem_terrain"] = True
    obj["dem_method"] = "raw_mesh"
    obj["dem_elev_min"] = float(elev_min)
    obj["dem_elev_max"] = float(elev_max)
    obj["dem_scale"] = scale
    # Projection origin used to place the raw-mesh vertices. Required
    # by the basemap UV unwrapper (see _terrain_xy_to_latlon_factory).
    obj["dem_center_lat"] = float(center_y)
    obj["dem_center_lon"] = float(center_x)
    
    if georaster.bounds:
        obj["dem_bounds_north"] = georaster.bounds['north']
        obj["dem_bounds_south"] = georaster.bounds['south']
        obj["dem_bounds_east"] = georaster.bounds['east']
        obj["dem_bounds_west"] = georaster.bounds['west']
    
    log(f"Raw mesh created: {len(verts)} vertices, {len(faces)} faces")
    log(f"Elevation range: {elev_min:.1f} - {elev_max:.1f} m")
    
    return obj


def apply_elevation_material(obj, style='ELEVATION'):
    """
    Apply a material to DEM terrain based on elevation.
    
    Args:
        obj: Blender mesh object
        style: Material style ('ELEVATION', 'GRAYSCALE')
    
    Returns:
        Material
    """
    if obj is None:
        return None
    
    mat_name = f"{obj.name}_Material"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    # Output
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    # BSDF
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (100, 0)
    bsdf.inputs['Roughness'].default_value = 0.8
    
    if style == 'ELEVATION':
        # Color ramp based on elevation attribute
        ramp = nodes.new(type='ShaderNodeValToRGB')
        ramp.location = (-200, 0)
        
        # Terrain colors
        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (0.2, 0.4, 0.15, 1)  # Green low
        
        elem_mid = ramp.color_ramp.elements.new(0.4)
        elem_mid.color = (0.6, 0.5, 0.3, 1)  # Brown mid
        
        ramp.color_ramp.elements[1].position = 1.0
        ramp.color_ramp.elements[1].color = (0.9, 0.9, 0.9, 1)  # White high
        
        # Normalize elevation
        map_range = nodes.new(type='ShaderNodeMapRange')
        map_range.location = (-400, 0)
        
        elev_min = obj.get("dem_elev_min", 0)
        elev_max = obj.get("dem_elev_max", 1000)
        
        map_range.inputs['From Min'].default_value = elev_min
        map_range.inputs['From Max'].default_value = elev_max
        
        # Elevation attribute
        attr = nodes.new(type='ShaderNodeAttribute')
        attr.location = (-600, 0)
        attr.attribute_name = "elevation"
        
        links.new(attr.outputs['Fac'], map_range.inputs['Value'])
        links.new(map_range.outputs['Result'], ramp.inputs['Fac'])
        links.new(ramp.outputs['Color'], bsdf.inputs['Base Color'])
    
    else:
        bsdf.inputs['Base Color'].default_value = (0.5, 0.5, 0.5, 1)
    
    links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])
    
    if obj.data.materials:
        obj.data.materials[0] = mat
    else:
        obj.data.materials.append(mat)
    
    return mat


def apply_georaster_elevations_to_graph(osmnx_obj, georaster, vertical_scale=1.0, vertical_offset=0.0):
    """Apply DEM elevations from a :class:`GeoRaster` to every mesh
    vertex of an OSMnx network, using the **mesh** as the source of
    truth for vertex positions.

    The mesh is what Blender renders, so sampling the DEM at each
    vertex's true ``(lat, lon)`` (computed by inverting the
    equirectangular-local projection that built the mesh) keeps the
    network glued to the terrain regardless of:

    * graph cache mismatches (simplification, conversion, projection),
    * curve-point ordering vs. ``nodes_data``,
    * whether the cached graph is in degrees (WGS84) or metres (UTM).

    The previous implementation sampled per *graph node* using the
    cached coordinates and then BFS-interpolated curve points, which
    fell back to ``(min+max)/2`` for orphan vertices and produced the
    notorious vertical spikes at high-relief locations.
    """
    import math
    import numpy as np

    if osmnx_obj is None or georaster is None:
        return False

    mesh = osmnx_obj.data
    # Both names are written depending on which import path created the
    # object; read either.
    scale = osmnx_obj.get("osmnx_scale")
    if scale is None:
        scale = osmnx_obj.get("scale", 0.001)

    stats = georaster.get_statistics()
    if stats is None:
        log("Could not get elevation statistics")
        return False
    min_elev = float(stats['min'])
    max_elev = float(stats['max'])

    # Small lift so the network sits visibly above the terrain mesh.
    graph_offset = 0.5

    log(
        f"Applying elevations to graph: {min_elev:.1f} - {max_elev:.1f} m "
        f"(with {graph_offset}m offset)"
    )

    # Single source of truth for the projection: the addon stores the
    # equirectangular-local origin used to build the mesh on the object
    # itself.
    center_lat = osmnx_obj.get("osmnx_center_lat")
    center_lon = osmnx_obj.get("osmnx_center_lon")
    if center_lat is None or center_lon is None:
        log(
            "Network has no osmnx_center_lat/lon; cannot reproject "
            "vertices to lat/lon. Re-import the network."
        )
        return False
    center_lat = float(center_lat)
    center_lon = float(center_lon)

    earth_radius = 6_371_000.0
    cos_lat = math.cos(math.radians(center_lat))
    if abs(cos_lat) < 1e-9:
        cos_lat = 1.0
    inv_scale = 1.0 / float(scale) if scale else 1.0
    meters_per_deg_lat = (math.pi / 180.0) * earth_radius
    meters_per_deg_lon = meters_per_deg_lat * cos_lat

    vertex_elevations = [0.0] * len(mesh.vertices)
    out_of_raster = 0
    for vert_idx, vert in enumerate(mesh.vertices):
        x_m = vert.co.x * inv_scale
        y_m = vert.co.y * inv_scale
        lat = center_lat + (y_m / meters_per_deg_lat)
        lon = center_lon + (x_m / meters_per_deg_lon)

        elev = georaster.get_elevation_at(lon, lat)
        if elev is None or (isinstance(elev, float) and np.isnan(elev)):
            out_of_raster += 1
            elev = min_elev

        elev = float(elev)
        vertex_elevations[vert_idx] = elev
        z = ((elev - min_elev + graph_offset) + vertical_offset) * scale * vertical_scale
        vert.co.z = z

    if out_of_raster:
        pct = 100.0 * out_of_raster / max(len(mesh.vertices), 1)
        log(
            f"  Warning: {out_of_raster} ({pct:.1f}%) of vertices are "
            "outside the DEM raster and were clamped to min elevation. "
            "Increase 'Padding' before fetching the DEM if this is large."
        )

    # Persist a per-vertex elevation attribute for downstream use
    # (Apply 3D, basemap, gradients).
    attr_name = "elevation"
    if attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name])
    elev_attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
    elev_attr.data.foreach_set("value", vertex_elevations)

    # Mirror per-node elevations into the cached graph too, so existing
    # OSMnx-side analyses (grades, GraphML) keep working.
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

    # Update the public elevation badges so the UI shows a meaningful
    # range (otherwise the panel would still display the previous run).
    osmnx_obj["osmnx_has_elevation"] = True
    osmnx_obj["osmnx_3d_applied"] = True
    osmnx_obj["osmnx_elev_scale_used"] = vertical_scale
    osmnx_obj["osmnx_elev_min"] = min_elev
    osmnx_obj["osmnx_elev_max"] = max_elev
    osmnx_obj["osmnx_elev_range"] = max_elev - min_elev

    log(
        f"Applied DEM elevations to {len(mesh.vertices)} vertices "
        f"(range: {max_elev - min_elev:.1f}m)"
    )
    return True

