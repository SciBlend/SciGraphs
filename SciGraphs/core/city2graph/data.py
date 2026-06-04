from ...utils.logger import log
from ..feature_tags import resolve_feature_tags, overture_type_from_preset, overture_place_keywords
from .get_c2g import get_city2graph
from . import utils
from . import overture_api


# OSMnx fallback tag dictionaries for feature types that the Overture
# REST API does not (yet) serve. Lifted from the recommendations of
# the city2graph notebooks (proximity.txt, examples.txt) where the
# authors switch to OSMnx for segments / land / water rather than
# fighting the partial REST coverage.
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
    """Tune ``ox.settings`` so feature/area queries behave reasonably.

    The default ``max_query_area_size`` is too restrictive for the
    bbox sizes a typical urban analysis uses (40 km² triggers
    thousands of Overpass sub-queries which can take minutes to
    complete). We raise it unconditionally to a generous value that
    still keeps accidental country-wide queries safe.
    """
    if ox is None:
        log("[c2g osmnx-config] OSMnx unavailable; cannot tune settings")
        return
    try:
        prev_mqas = getattr(ox.settings, "max_query_area_size", None)
        prev_timeout = getattr(ox.settings, "timeout", None)
        # 200 km × 200 km — comfortably covers every city scale we
        # support. Beyond this the user almost certainly meant to set
        # a smaller bbox.
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
    """Fallback path for feature types Overture REST cannot serve.

    Uses OSMnx ``features_from_bbox`` for polygons/lines (buildings,
    water, land) and ``graph_from_bbox`` + ``c2g.nx_to_gdf`` for
    segments — the same combos used in the c2g notebooks. Returns a
    GeoDataFrame or ``None`` if OSMnx itself cannot be imported.
    """
    from ..osmnx import features as ox_features
    from ..osmnx.get_osmnx import get_osmnx

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
            # NOTE: do not simplify here. Simplification collapses
            # consecutive intersections into a single edge whose
            # geometry attribute may then be missing or reduced to a
            # Point representative — which is what made the rendered
            # segments look like points instead of street lines.
            graph = _osmnx_graph_from_bbox(
                ox, n, s, e, w, network_type='all', simplify=False, retain_all=True,
            )
            if graph is None or len(graph.edges) == 0:
                log("[c2g fallback] No segments returned by OSMnx for this bbox")
                return None
            # Prefer OSMnx itself: ``graph_to_gdfs(edges=True)`` is
            # guaranteed to return a GeoDataFrame whose geometry
            # column holds LineStrings (with the real polyline of
            # each street, not just the (u, v) endpoints). The
            # equivalent ``c2g.nx_to_gdf`` route can drop the
            # geometry to a centroid for some edge types, which then
            # materialised as one isolated vertex per edge in
            # Blender (the symptom you saw).
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
    # OSMnx returns mixed (Point/LineString/Polygon) for some categories.
    # For Buildings/Land/Water we want polygons (and their multi-variants).
    try:
        if feature_type in ('building', 'land', 'water'):
            polygonal = gdf[gdf.geometry.geom_type.isin(['Polygon', 'MultiPolygon'])].copy()
            if len(polygonal) > 0:
                gdf = polygonal
    except Exception:  # noqa: BLE001 — defensive: keep raw if filter explodes
        pass
    log(f"[c2g fallback] OSMnx returned {len(gdf)} {feature_type} features")
    return gdf


def get_boundaries(place_name, user_agent="scigraphs"):
    """
    Retrieve polygon boundary for a place using Nominatim geocoding.
    
    New in city2graph 0.3.1.
    
    Args:
        place_name: Name of the place to geocode (e.g., "Liverpool, UK")
        user_agent: User agent string for Nominatim API
    
    Returns:
        GeoDataFrame with polygon geometry and place_name property
    """
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
    """
    Download Overture Maps data using city2graph's native API.
    
    This uses city2graph 0.3.1+ which calls the overturemaps CLI directly.
    
    Args:
        place_name: Name of place to geocode (e.g., "Liverpool, UK")
        bbox: Alternative - bounding box as [min_lon, min_lat, max_lon, max_lat]
        types: List of feature types to download
        osmnx_obj: Optional OSMnx object to align coordinates
    
    Returns:
        dict: Dictionary mapping feature type to GeoDataFrame
    """
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
    """
    Process segments from Overture Maps to be split by connectors and extract barriers.
    
    New in city2graph 0.3.1.
    
    Args:
        segments_gdf: GeoDataFrame containing road segments
        connectors_gdf: Optional GeoDataFrame with connector points
        get_barriers: Whether to generate barrier geometries from level rules
        threshold: Distance threshold for endpoint clustering
    
    Returns:
        GeoDataFrame: Processed segments with additional columns
    """
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


def load_overture_data(bbox=None, types=None, osmnx_obj=None, use_city2graph_api=False, place_name=None, limit=10000, place_categories=None):
    """
    Download Overture Maps data within bounding box.
    
    Args:
        bbox: Tuple of (north, south, east, west) coordinates
        types: List of feature types to download (building, segment, place, water, land)
        osmnx_obj: Optional OSMnx object to align coordinates
        use_city2graph_api: If True, use city2graph's native API (requires CLI)
        place_name: Place name to geocode (alternative to bbox, new in 0.3.1)
    
    Returns:
        dict: Dictionary mapping feature type to created objects
    """
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
            # Overture REST is the preferred source where available
            # (richest schema, best for buildings/places). For
            # everything else — and as a graceful fallback when
            # Overture returns nothing — use OSMnx, which is what the
            # city2graph notebooks themselves do (see proximity.txt
            # and examples.txt).
            if feature_type == 'building':
                gdf = overture_api.query_overture_buildings(bbox, limit=limit)
                if gdf is None or len(gdf) == 0:
                    log("Buildings not returned by Overture; falling back to OSMnx")
                    gdf = _fetch_features_via_osmnx(bbox, 'building')
            elif feature_type == 'place':
                gdf = overture_api.query_overture_places(bbox, categories=place_categories, limit=limit)
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
    """Keep only OSM node elements from a features GeoDataFrame.

    Mirrors the notebook workflow (``element == "node"``) so point-feature
    counts match. Returns the gdf unchanged when no element index is present.
    """
    if gdf is None or not hasattr(gdf.index, "names"):
        return gdf
    for name in gdf.index.names:
        if name and "element" in name.lower():
            return gdf[gdf.index.get_level_values(name) == "node"]
    return gdf


def download_features(bbox, source, feature_type, custom_tags="", osmnx_obj=None,
                      limit=10000, nodes_only=False, place_name=None):
    """Download and materialise features using the shared import selector.

    When ``source`` is OSMnx and ``place_name`` is provided, the features are
    queried within the place's administrative polygon (``features_from_place``)
    instead of the bounding box, matching the notebook workflow. Otherwise the
    bounding box is used.
    """
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
            )
            if result:
                for objects in result.values():
                    for obj in objects:
                        obj["feature_source"] = "OVERTURE"
                        obj["feature_type"] = feature_type
            return result
        log("Custom tags are not available in Overture mode; using OSMnx")

    from ..osmnx import features as ox_features
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
    """
    Download building footprints from Overture Maps.
    
    Args:
        bbox: Tuple of (north, south, east, west) coordinates
        osmnx_obj: Optional OSMnx object to align coordinates
    
    Returns:
        list: Created Blender objects
    """
    result = load_overture_data(bbox, types=['building'], osmnx_obj=osmnx_obj)
    if result and 'building' in result:
        return result['building']
    return []


def load_overture_roads(bbox, osmnx_obj=None):
    """
    Download road segments from Overture Maps.
    
    Args:
        bbox: Tuple of (north, south, east, west) coordinates
        osmnx_obj: Optional OSMnx object to align coordinates
    
    Returns:
        list: Created Blender objects
    """
    result = load_overture_data(bbox, types=['segment'], osmnx_obj=osmnx_obj)
    if result and 'segment' in result:
        return result['segment']
    return []


def load_overture_places(bbox, osmnx_obj=None):
    """
    Download places (POIs) from Overture Maps.
    
    Args:
        bbox: Tuple of (north, south, east, west) coordinates
        osmnx_obj: Optional OSMnx object to align coordinates
    
    Returns:
        list: Created Blender objects
    """
    result = load_overture_data(bbox, types=['place'], osmnx_obj=osmnx_obj)
    if result and 'place' in result:
        return result['place']
    return []


def load_data_from_file(filepath, osmnx_obj=None):
    """
    Load urban data from file (GeoJSON, Shapefile, etc.).
    
    Args:
        filepath: Path to file
        osmnx_obj: Optional OSMnx object to align coordinates
    
    Returns:
        list: Created Blender objects
    """
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

