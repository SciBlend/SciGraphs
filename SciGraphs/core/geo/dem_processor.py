# Elevation processing: nodata filling, normalization, raster-to-mesh.

import bpy
import bmesh
import numpy as np
from scigraphs_core.logger import log


def fill_nodata(data, nodata_value=None, method='nearest'):
    """Fill NoData holes in an elevation array by interpolation; nodata_value=None auto-detects NaN and inf, and without scipy both methods degrade to the mean."""
    result = data.copy().astype(np.float32)
    
    if nodata_value is not None:
        invalid_mask = (result == nodata_value)
    else:
        invalid_mask = np.isnan(result) | np.isinf(result)
    
    invalid_count = np.sum(invalid_mask)
    if invalid_count == 0:
        return result
    
    log(f"Filling {invalid_count} NoData values...")
    
    result[invalid_mask] = np.nan
    
    valid_mask = ~np.isnan(result)
    if not np.any(valid_mask):
        log("Warning: No valid data found")
        return np.zeros_like(result)
    
    if method == 'nearest':
        try:
            from scipy import ndimage

            indices = ndimage.distance_transform_edt(
                np.isnan(result),
                return_distances=False,
                return_indices=True
            )
            result = result[indices[0], indices[1]]

        except ImportError:
            mean_val = np.nanmean(result)
            result[np.isnan(result)] = mean_val

    elif method == 'linear':
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
    """Rescale an elevation array to target_range as float32; uint8 is assumed to span the full 0-255 range, every other dtype uses its own min and max."""
    data_f = data.astype(np.float32)

    if 'uint8' in dtype:
        data_min, data_max = 0, 255
    elif 'int16' in dtype:
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

    t_min, t_max = target_range
    normalized = (data_f - data_min) / (data_max - data_min)
    normalized = normalized * (t_max - t_min) + t_min
    
    return normalized


def calculate_displace_strength(georaster, scale=1.0):
    """Return (strength, midlevel) for a Displace modifier: one unit of displacement is one meter of elevation, so strength is the elevation range times scale (0.001 for meters to Blender km units)."""
    stats = georaster.get_statistics()
    if stats is None:
        return 1.0, 0.5
    
    elev_min = stats['min']
    elev_max = stats['max']
    elev_range = elev_max - elev_min
    
    dtype = georaster.dtype

    if 'float' in dtype:
        strength = elev_range * scale
        midlevel = (elev_min - elev_min) / elev_range if elev_range > 0 else 0

    elif 'int16' in dtype:
        # SRTM ships int16 meters.
        strength = elev_range * scale
        midlevel = 0

    elif 'uint16' in dtype:
        strength = elev_range * scale
        midlevel = 0

    elif 'uint8' in dtype:
        # A range under 10 means normalized, not meters, so use the 0-255 span.
        if elev_range > 10:
            strength = elev_range * scale
        else:
            strength = 255 * scale
        midlevel = 0


    else:
        strength = elev_range * scale
        midlevel = 0
    
    log(f"Displace strength: {strength:.4f}, midlevel: {midlevel:.4f}")
    log(f"Elevation range: {elev_min:.1f} - {elev_max:.1f} m ({elev_range:.1f} m)")
    
    return strength, midlevel


def create_raster_extent_mesh(georaster, scale=0.001, name="DEM_Plane", osmnx_obj=None):
    """Create a flat quad covering the raster extent, ready for displacement. Pass osmnx_obj to reuse that network's projection origin, or plane and network land in different places."""
    bounds = georaster.bounds
    if bounds is None:
        log("GeoRaster has no bounds")
        return None
    
    raster_center_lon = (bounds['east'] + bounds['west']) / 2
    raster_center_lat = (bounds['north'] + bounds['south']) / 2
    
    graph_center_lon = raster_center_lon
    graph_center_lat = raster_center_lat
    
    if osmnx_obj is not None and osmnx_obj.get("is_osmnx", False):
        stored_lat = osmnx_obj.get("osmnx_center_lat")
        stored_lon = osmnx_obj.get("osmnx_center_lon")
        
        if stored_lat is not None and stored_lon is not None:
            graph_center_lat = stored_lat
            graph_center_lon = stored_lon
    
    if georaster.is_geographic_crs():
        lat_m = 111000  # meters per degree latitude
        lon_m = 111000 * np.cos(np.radians(graph_center_lat))
        
        width_m = (bounds['east'] - bounds['west']) * lon_m
        height_m = (bounds['north'] - bounds['south']) * lat_m

        offset_x_m = (raster_center_lon - graph_center_lon) * lon_m
        offset_y_m = (raster_center_lat - graph_center_lat) * lat_m
    else:
        width_m = bounds['east'] - bounds['west']
        height_m = bounds['north'] - bounds['south']
        offset_x_m = 0
        offset_y_m = 0
    
    width = width_m * scale
    height = height_m * scale
    offset_x = offset_x_m * scale
    offset_y = offset_y_m * scale
    
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
    
    mesh.uv_layers.new(name="UVMap")
    uv_layer = mesh.uv_layers.active.data
    uv_coords = [(0, 0), (1, 0), (1, 1), (0, 1)]
    for i, uv in enumerate(uv_coords):
        uv_layer[i].uv = uv
    
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    obj["is_dem_terrain"] = True
    obj["dem_width_m"] = width_m
    obj["dem_height_m"] = height_m
    obj["dem_scale"] = scale
    # Origin of the local frame; the basemap UV unwrapper inverts XY through it.
    obj["dem_center_lat"] = float(graph_center_lat)
    obj["dem_center_lon"] = float(graph_center_lon)
    obj["dem_bounds_north"] = bounds['north']
    obj["dem_bounds_south"] = bounds['south']
    obj["dem_bounds_east"] = bounds['east']
    obj["dem_bounds_west"] = bounds['west']
    
    return obj


def apply_displace_modifier(obj, georaster, subdivision_levels=6, scale=0.001):
    """Turn a flat plane into terrain with Subdivision plus Displace modifiers, the fast path. The DEM image is loaded as Non-Color: sRGB would gamma correct the elevations and flatten the relief."""
    if obj is None or georaster is None:
        return False
    
    try:
        img = bpy.data.images.load(georaster.filepath)
        img.colorspace_settings.name = 'Non-Color'
    except Exception as e:
        log(f"Error loading DEM as image: {e}")
        return False
    
    tex_name = f"{obj.name}_DEM_Tex"
    tex = bpy.data.textures.new(tex_name, type='IMAGE')
    tex.image = img
    tex.extension = 'EXTEND'
    
    subsurf = obj.modifiers.new(name="Subdivision", type='SUBSURF')
    subsurf.subdivision_type = 'SIMPLE'
    subsurf.levels = subdivision_levels
    subsurf.render_levels = subdivision_levels
    
    displace = obj.modifiers.new(name="DEM_Displace", type='DISPLACE')
    displace.texture = tex
    displace.texture_coords = 'UV'
    displace.direction = 'Z'
    
    strength, midlevel = calculate_displace_strength(georaster, scale)
    displace.strength = strength
    displace.mid_level = midlevel
    
    obj["dem_texture"] = tex_name
    obj["dem_image"] = img.name
    
    log(f"Applied Displace modifier with {subdivision_levels} subdivisions")
    
    return True


def raster_to_mesh(georaster, scale=0.001, subsample=1, name="DEM_RawMesh",
                   osmnx_obj=None, vertical_scale=1.0, vertical_offset=0.0):
    """Build a mesh with one vertex per raster pixel, Z from elevation: slower than the Displace path but exact. subsample=2 halves each axis; vertical_offset is meters, applied before scaling."""
    if georaster is None or georaster.data is None:
        return None
    
    data = georaster.data
    
    if subsample > 1:
        data = data[::subsample, ::subsample]
    
    height, width = data.shape
    log(f"Building raw mesh: {width}x{height} vertices")
    
    data = fill_nodata(data, georaster.nodata)
    
    if georaster.geotransform:
        ox, sx, _, oy, _, sy = georaster.geotransform
        sx *= subsample
        sy *= subsample
    else:
        # No geotransform: one unit per pixel, Y flipped like a raster.
        ox, oy = 0, 0
        sx, sy = 1, -1

    # Reuse the network's projection origin so terrain and network share a frame.
    if osmnx_obj is not None and osmnx_obj.get("is_osmnx", False):
        center_y = osmnx_obj.get("osmnx_center_lat")
        center_x = osmnx_obj.get("osmnx_center_lon")

        if center_y is not None and center_x is not None:
            log(f"Using stored OSMnx center: {center_y:.5f}, {center_x:.5f}")
        else:
            center_x = (georaster.bounds['east'] + georaster.bounds['west']) / 2
            center_y = (georaster.bounds['north'] + georaster.bounds['south']) / 2
    else:
        center_x = (georaster.bounds['east'] + georaster.bounds['west']) / 2
        center_y = (georaster.bounds['north'] + georaster.bounds['south']) / 2
    
    if georaster.is_geographic_crs():
        lat_m = 111000  # meters per degree latitude
        lon_m = 111000 * np.cos(np.radians(center_y))
    else:
        lat_m = 1
        lon_m = 1

    elev_min = np.nanmin(data)
    elev_max = np.nanmax(data)
    
    verts = []
    for row in range(height):
        for col in range(width):
            if georaster.is_geographic_crs():
                geo_x = ox + col * sx
                geo_y = oy + row * sy
                
                x_m = (geo_x - center_x) * lon_m
                y_m = (geo_y - center_y) * lat_m
            else:
                x_m = ox + col * sx - center_x
                y_m = oy + row * sy - center_y
            
            z_m = float(data[row, col])

            # Must match apply_georaster_elevations_to_graph, or the network floats.
            x = x_m * scale
            y = y_m * scale
            z = ((z_m - elev_min) + vertical_offset) * scale * vertical_scale
            
            verts.append((x, y, z))
    
    faces = []
    for row in range(height - 1):
        for col in range(width - 1):
            v0 = row * width + col
            v1 = v0 + 1
            v2 = v0 + width + 1
            v3 = v0 + width
            faces.append((v0, v1, v2, v3))
    
    mesh = bpy.data.meshes.new(f"{name}_Mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    for polygon in mesh.polygons:
        polygon.use_smooth = True
    
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    
    if "elevation" in mesh.attributes:
        mesh.attributes.remove(mesh.attributes["elevation"])
    
    elev_attr = mesh.attributes.new(name="elevation", type='FLOAT', domain='POINT')
    elevations = data.flatten().astype(np.float32)
    elev_attr.data.foreach_set("value", elevations)
    
    obj["is_dem_terrain"] = True
    obj["dem_method"] = "raw_mesh"
    obj["dem_elev_min"] = float(elev_min)
    obj["dem_elev_max"] = float(elev_max)
    obj["dem_scale"] = scale
    # Projection origin these vertices were placed against, read back by the
    # basemap UV unwrapper.
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
    """Give terrain a material driven by the "elevation" vertex attribute: style 'ELEVATION' ramps green to brown to white across the stored elevation range, anything else gets flat gray."""
    if obj is None:
        return None
    
    mat_name = f"{obj.name}_Material"
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()
    
    output = nodes.new(type='ShaderNodeOutputMaterial')
    output.location = (400, 0)
    
    bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
    bsdf.location = (100, 0)
    bsdf.inputs['Roughness'].default_value = 0.8
    
    if style == 'ELEVATION':
        ramp = nodes.new(type='ShaderNodeValToRGB')
        ramp.location = (-200, 0)

        ramp.color_ramp.elements[0].position = 0.0
        ramp.color_ramp.elements[0].color = (0.2, 0.4, 0.15, 1)

        elem_mid = ramp.color_ramp.elements.new(0.4)
        elem_mid.color = (0.6, 0.5, 0.3, 1)

        ramp.color_ramp.elements[1].position = 1.0
        ramp.color_ramp.elements[1].color = (0.9, 0.9, 0.9, 1)

        map_range = nodes.new(type='ShaderNodeMapRange')
        map_range.location = (-400, 0)
        
        elev_min = obj.get("dem_elev_min", 0)
        elev_max = obj.get("dem_elev_max", 1000)
        
        map_range.inputs['From Min'].default_value = elev_min
        map_range.inputs['From Max'].default_value = elev_max
        
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
    """Lift every mesh vertex of an OSMnx network onto the DEM surface.

    Sampled per mesh vertex, at the (lat, lon) recovered by inverting the
    projection that built the mesh. Going through the cached graph instead gives
    vertical spikes in high relief: its node coordinates may be simplified,
    projected to UTM or reordered, and orphans land at the midpoint elevation."""
    import math
    import numpy as np

    if osmnx_obj is None or georaster is None:
        return False

    mesh = osmnx_obj.data
    # Import paths disagree on the key name, so try both.
    scale = osmnx_obj.get("osmnx_scale")
    if scale is None:
        scale = osmnx_obj.get("scale", 0.001)

    stats = georaster.get_statistics()
    if stats is None:
        log("Could not get elevation statistics")
        return False
    min_elev = float(stats['min'])
    max_elev = float(stats['max'])

    # Meters of lift so the network reads as above the terrain, not inside it.
    graph_offset = 0.5

    log(
        f"Applying elevations to graph: {min_elev:.1f} - {max_elev:.1f} m "
        f"(with {graph_offset}m offset)"
    )

    # The origin lives on the object; recomputing it here would drift.
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

    # Apply 3D, basemap and gradients all read this attribute back.
    attr_name = "elevation"
    if attr_name in mesh.attributes:
        mesh.attributes.remove(mesh.attributes[attr_name])
    elev_attr = mesh.attributes.new(name=attr_name, type='FLOAT', domain='POINT')
    elev_attr.data.foreach_set("value", vertex_elevations)

    # Mirror back into the cached graph for grade calculations and GraphML export.
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

    # Without this the panel keeps showing the previous run's range.
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

