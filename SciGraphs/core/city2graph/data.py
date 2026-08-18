"""Download urban features from Overture Maps or OSMnx and build Blender objects. The city2graph calls need version 0.3.1 or newer and return None when it is missing."""
from scigraphs_core.logger import log
from scigraphs_core.feature_tags import resolve_feature_tags, overture_type_from_preset, overture_place_keywords
from scigraphs_core.city2graph.get_c2g import get_city2graph
from . import utils
from scigraphs_core.city2graph import overture_api


# OSMnx tag sets for the feature types Overture REST does not serve.
_OSMNX_FALLBACK_TAGS = {
    'building': {"building": True},
    'water': {
        "natural": ["water", "wetland", "bay", "strait", "coastline"],
        "waterway": True,
    },
    'land': {
        "landuse": True,
        "leisure": ["park", "garden", "nature_reserve", "playground"],
        "natural": ["wood", "scrub", "grassland", "heath", "beach"],
    },
}


def _configure_osmnx_for_features(ox):
    """Raise ``ox.settings.max_query_area_size`` and the timeout: the 40 km² default splits a typical urban bbox into thousands of Overpass sub-queries taking minutes."""
    if ox is None:
        log("[c2g osmnx-config] OSMnx unavailable; cannot tune settings")
        return
    try:
        prev_mqas = getattr(ox.settings, "max_query_area_size", None)
        prev_timeout = getattr(ox.settings, "timeout", None)
        # 200 km x 200 km covers every city scale.
        new_mqas = 200 * 1000 * 200 * 1000  # 4e10 m²
        ox.settings.max_query_area_size = new_mqas
        ox.settings.use_cache = True
        ox.settings.timeout = max(prev_timeout or 0, 300)
        log(
            f"[c2g osmnx-config] max_query_area_size: "
            f"{prev_mqas} → {ox.settings.max_query_area_size}; "
            f"timeout: {prev_timeout} → {ox.settings.timeout}; "
            f"use_cache=True"
        )
    except AttributeError as exc:
        # Older OSMnx: settings live in a different namespace.
        log(f"[c2g osmnx-config] could not tune ox.settings: {exc}")


def _fetch_features_via_osmnx(bbox, feature_type):
    """Fetch a feature type from OSMnx when Overture REST cannot serve it: buildings, water and land from ``features_from_bbox``, segments from a bbox graph. None means OSMnx is missing or the area holds nothing."""
    from scigraphs_core.osmnx import features as ox_features
    from scigraphs_core.osmnx.get_osmnx import get_osmnx

    log(f"[c2g fallback] entering OSMnx path for type='{feature_type}', bbox={bbox}")
    _configure_osmnx_for_features(get_osmnx())

    if feature_type == 'segment':
        try:
            from ..data_io.importer import _osmnx_graph_from_bbox
            ox = get_osmnx()
            if ox is None:
                log("[c2g fallback] OSMnx not available; cannot fetch segments")
                return None
            n, s, e, w = bbox
            # Do not simplify: collapsing consecutive intersections can shrink an
            # edge geometry to a Point, and segments then render as points.
            graph = _osmnx_graph_from_bbox(
                ox, n, s, e, w, network_type='all', simplify=False, retain_all=True,
            )
            if graph is None or len(graph.edges) == 0:
                log("[c2g fallback] No segments returned by OSMnx for this bbox")
                return None
            # ``graph_to_gdfs(edges=True)`` keeps each street's real polyline;
            # ``c2g.nx_to_gdf`` drops some edges to a centroid, one vertex each.
            edges_gdf = ox.graph_to_gdfs(graph, nodes=False, edges=True)
            try:
                geom_types = edges_gdf.geometry.geom_type.value_counts().to_dict()
                log(f"[c2g fallback] segments geometry types: {geom_types}")
            except Exception:  # noqa: BLE001
                pass
            log(f"[c2g fallback] OSMnx returned {len(edges_gdf)} road segments")
            return edges_gdf
        except Exception as exc:  # pragma: no cover - upstream-dependent
            log(f"[c2g fallback] segment fetch failed: {exc}")
            import traceback
            traceback.print_exc()
            return None

    tags = _OSMNX_FALLBACK_TAGS.get(feature_type)
    if tags is None:
        return None
    gdf = ox_features.features_from_bbox(bbox, tags)
    if gdf is None:
        return None
    # OSMnx mixes geometry types per category; these three want the polygons.
    try:
        if feature_type in ('building', 'land', 'water'):
            polygonal = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
            if len(polygonal) > 0:
                gdf = polygonal
    except Exception:  # noqa: BLE001 - defensive: keep raw if filter explodes
        pass
    log(f"[c2g fallback] OSMnx returned {len(gdf)} {feature_type} features")
    return gdf


def get_boundaries(place_name, user_agent="scigraphs"):
    """Geocode ``place_name`` with Nominatim and return its boundary polygon."""
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.data import get_boundaries as c2g_get_boundaries
        
        log(f"Geocoding boundaries for '{place_name}'...")
        boundary_gdf = c2g_get_boundaries(place_name, user_agent=user_agent)
        
        if boundary_gdf is None or len(boundary_gdf) == 0:
            log(f"No boundary found for '{place_name}'")
            return None
        
        log(f"Found boundary for '{place_name}'")
        return boundary_gdf
        
    except Exception as e:
        log(f"Error geocoding boundaries: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_overture_data_c2g(place_name=None, bbox=None, types=None, osmnx_obj=None):
    """Download Overture Maps data through city2graph, which shells out to the overturemaps CLI. Pass ``place_name`` or ``bbox`` as [min_lon, min_lat, max_lon, max_lat]. Returns feature type to GeoDataFrame."""
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.data import load_overture_data as c2g_load_overture
        
        if place_name:
            log(f"Downloading Overture Maps data for '{place_name}'...")
            data = c2g_load_overture(
                place_name=place_name,
                types=types,
                save_to_file=False,
                return_data=True
            )
        elif bbox:
            log(f"Downloading Overture Maps data for bbox: {bbox}...")
            data = c2g_load_overture(
                area=bbox,
                types=types,
                save_to_file=False,
                return_data=True
            )
        else:
            log("Either place_name or bbox must be provided")
            return None
        
        if data is None or len(data) == 0:
            log("No data downloaded from Overture Maps")
            return None
        
        for dtype, gdf in data.items():
            log(f"Downloaded {len(gdf)} {dtype} features")
        
        return data
        
    except Exception as e:
        log(f"Error downloading Overture data via city2graph: {e}")
        import traceback
        traceback.print_exc()
        return None


def process_overture_segments(segments_gdf, connectors_gdf=None, get_barriers=True, threshold=1.0):
    """Split Overture segments at their connectors and extract barriers from the segments' level rules; ``threshold`` is the endpoint clustering distance."""
    c2g = get_city2graph()
    if c2g is None:
        log("city2graph is not available")
        return None
    
    try:
        from city2graph.data import process_overture_segments as c2g_process_segments
        
        log("Processing Overture segments...")
        processed = c2g_process_segments(
            segments_gdf,
            get_barriers=get_barriers,
            connectors_gdf=connectors_gdf,
            threshold=threshold
        )
        
        if processed is None or len(processed) == 0:
            log("No processed segments returned")
            return None
        
        log(f"Processed {len(processed)} segments")
        return processed
        
    except Exception as e:
        log(f"Error processing Overture segments: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_overture_data(bbox=None, types=None, osmnx_obj=None, use_city2graph_api=False, place_name=None, limit=10000, place_categories=None, overture_api_key=None):
    """Download Overture Maps features and build Blender objects from them.

    ``bbox`` is (north, south, east, west). Building and place come from the
    Overture REST API; segment, water and land fall through to OSMnx. Setting
    ``use_city2graph_api`` or a ``place_name`` switches to the CLI downloader.
    ``overture_api_key`` is forwarded to the REST client, which cannot read a
    Blender preference; None falls back to OVERTURE_API_KEY, then the demo key."""
    if use_city2graph_api or place_name:
        if place_name:
            data = load_overture_data_c2g(place_name=place_name, types=types, osmnx_obj=osmnx_obj)
        elif bbox:
            north, south, east, west = bbox
            api_bbox = [west, south, east, north]
            data = load_overture_data_c2g(bbox=api_bbox, types=types, osmnx_obj=osmnx_obj)
        else:
            log("Either bbox or place_name must be provided")
            return None
        
        if data is None:
            return None
        
        result_objects = {}
        for feature_type, gdf in data.items():
            collection_name = f"C2G_Overture_{feature_type.capitalize()}"
            objects = utils.gdf_to_blender_mesh(
                gdf,
                name=f"{feature_type}_overture",
                collection_name=collection_name,
                osmnx_obj=osmnx_obj
            )
            if objects:
                result_objects[feature_type] = objects
        
        return result_objects if result_objects else None
    
    if bbox is None:
        log("bbox is required when not using city2graph API")
        return None
    
    if types is None:
        types = ['building', 'place']
    
    north, south, east, west = bbox
    log(f"Downloading Overture Maps data via REST API for bbox: N{north}, S{south}, E{east}, W{west}")
    log(f"Feature types: {types}")
    
    result_objects = {}
    
    for feature_type in types:
        try:
            log(f"Downloading {feature_type} features...")
            
            gdf = None
            # Overture REST has the richest schema for buildings and places;
            # everything else, and anything it returns empty, goes to OSMnx.
            if feature_type == 'building':
                gdf = overture_api.query_overture_buildings(bbox, limit=limit, api_key=overture_api_key)
                if gdf is None or len(gdf) == 0:
                    log("Buildings not returned by Overture; falling back to OSMnx")
                    gdf = _fetch_features_via_osmnx(bbox, 'building')
            elif feature_type == 'place':
                gdf = overture_api.query_overture_places(bbox, categories=place_categories, limit=limit, api_key=overture_api_key)
            elif feature_type in ('segment', 'water', 'land'):
                log(
                    f"'{feature_type}' is not served by Overture REST; "
                    "using OSMnx fallback (matches city2graph notebooks)"
                )
                gdf = _fetch_features_via_osmnx(bbox, feature_type)
            else:
                log(f"Unknown feature type: {feature_type}")
                continue

            if gdf is None or len(gdf) == 0:
                log(f"No {feature_type} features found")
                continue
            
            log(f"Downloaded {len(gdf)} {feature_type} features")
            
            collection_name = f"C2G_Overture_{feature_type.capitalize()}"
            objects = utils.gdf_to_blender_mesh(
                gdf,
                name=f"{feature_type}_overture",
                collection_name=collection_name,
                osmnx_obj=osmnx_obj
            )
            
            if objects:
                result_objects[feature_type] = objects
                log(f"Created {len(objects)} object(s) for {feature_type}")
            
        except Exception as e:
            log(f"Error downloading {feature_type}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if len(result_objects) == 0:
        log("No features were downloaded from Overture Maps API")
        log("")
        log("Possible causes:")
        log("1. No data coverage for this area")
        log("2. DEMO-API-KEY has limitations - get a production key")
        log("3. Area might be too large - try smaller bbox")
        log("")
        log("Alternative: Use OSMnx to download OpenStreetMap buildings:")
        log("- OSMnx > Download Features > Building Type")
        log("- Works for all areas globally with OSM data")
        return None
    
    return result_objects


def _filter_gdf_to_nodes(gdf):
    """Keep only OSM node elements, or the whole frame when its index has no element level."""
    if gdf is None or not hasattr(gdf.index, "names"):
        return gdf
    for name in gdf.index.names:
        if name and "element" in name.lower():
            return gdf[gdf.index.get_level_values(name) == "node"]
    return gdf


def download_features(bbox, source, feature_type, custom_tags="", osmnx_obj=None,
                      limit=10000, nodes_only=False, place_name=None,
                      overture_api_key=None):
    """Download features through the shared import selector and build objects. With ``source`` OSMnx and a ``place_name`` given, the query runs inside that place's administrative polygon rather than the bbox."""
    source = source or 'OVERTURE'
    feature_type = feature_type or 'BUILDING'

    if source == 'OVERTURE':
        overture_type = overture_type_from_preset(feature_type)
        if overture_type is not None:
            place_categories = (
                overture_place_keywords(feature_type)
                if overture_type == 'place' else None
            )
            result = load_overture_data(
                bbox=bbox,
                types=[overture_type],
                osmnx_obj=osmnx_obj,
                limit=limit,
                place_categories=place_categories,
                overture_api_key=overture_api_key,
            )
            if result:
                for objects in result.values():
                    for obj in objects:
                        obj["feature_source"] = "OVERTURE"
                        obj["feature_type"] = feature_type
            return result
        log("Custom tags are not available in Overture mode; using OSMnx")

    from scigraphs_core.osmnx import features as ox_features
    from types import SimpleNamespace

    tags = resolve_feature_tags(SimpleNamespace(
        feat_type=feature_type,
        feat_custom_tags=custom_tags,
    ))
    if not tags:
        return None

    if place_name:
        log(f"Querying OSMnx features within place polygon: '{place_name}'")
        gdf = ox_features.features_from_place(place_name, tags)
    else:
        gdf = ox_features.features_from_bbox(bbox, tags)
    if gdf is None or len(gdf) == 0:
        return None

    if nodes_only:
        gdf = _filter_gdf_to_nodes(gdf)
        log(f"Filtered to node elements: {len(gdf)} features")
        if len(gdf) == 0:
            return None

    collection_name = f"C2G_OSMnx_{feature_type.title().replace('_', '')}"
    objects = utils.gdf_to_blender_mesh(
        gdf,
        name=f"osmnx_{feature_type.lower()}",
        collection_name=collection_name,
        osmnx_obj=osmnx_obj,
    )
    if not objects:
        return None

    for obj in objects:
        obj["is_osm_features"] = True
        obj["feature_type"] = feature_type
        obj["feature_source"] = "OSMNX"

    return {feature_type.lower(): objects}


def load_overture_buildings(bbox, osmnx_obj=None):
    """Download building footprints from Overture Maps into Blender objects."""
    result = load_overture_data(bbox, types=['building'], osmnx_obj=osmnx_obj)
    if result and 'building' in result:
        return result['building']
    return []


def load_overture_roads(bbox, osmnx_obj=None):
    """Download road segments from Overture Maps into Blender objects."""
    result = load_overture_data(bbox, types=['segment'], osmnx_obj=osmnx_obj)
    if result and 'segment' in result:
        return result['segment']
    return []


def load_overture_places(bbox, osmnx_obj=None):
    """Download places (POIs) from Overture Maps into Blender objects."""
    result = load_overture_data(bbox, types=['place'], osmnx_obj=osmnx_obj)
    if result and 'place' in result:
        return result['place']
    return []


def load_data_from_file(filepath, osmnx_obj=None):
    """Load a geospatial file (GeoJSON, Shapefile, and the rest) into Blender objects."""
    try:
        import geopandas as gpd
        
        log(f"Loading data from {filepath}")
        gdf = gpd.read_file(filepath)
        
        if gdf is None or len(gdf) == 0:
            log("No data found in file")
            return []
        
        log(f"Loaded {len(gdf)} features from file")
        
        import os
        filename = os.path.basename(filepath).split('.')[0]
        
        objects = utils.gdf_to_blender_mesh(
            gdf,
            name=f"C2G_{filename}",
            collection_name="C2G_Imported",
            osmnx_obj=osmnx_obj
        )
        
        return objects
        
    except Exception as e:
        log(f"Error loading data from file: {e}")
        return []

