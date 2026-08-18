"""Metapath analysis for heterogeneous urban graphs: dual graphs from street
networks, amenities bridged to street segments, metapaths between amenities."""

def create_street_dual_graph_c2g(osmnx_graph):
    """Build the dual graph of a street network, with segments as nodes. city2graph
    returns a 3-level MultiIndex ``(x, y, nan)`` for the dual nodes; the third level
    is dropped on node and edge index alike, or nothing downstream can match up."""
    import pandas as pd
    import city2graph as c2g
    
    street_primary_nodes, street_primary_edges = c2g.nx_to_gdf(osmnx_graph)
    
    dual_nodes_gdf, dual_edges_gdf = c2g.dual_graph((street_primary_nodes, street_primary_edges))
    
    dual_nodes_gdf.geometry = dual_nodes_gdf.geometry.centroid
    
    if isinstance(dual_nodes_gdf.index, pd.MultiIndex) and dual_nodes_gdf.index.nlevels == 3:
        new_index = [(idx[0], idx[1]) for idx in dual_nodes_gdf.index]
        dual_nodes_gdf.index = pd.Index(new_index)

        new_edge_index = [
            ((idx[0][0], idx[0][1]), (idx[1][0], idx[1][1])) 
            if isinstance(idx, tuple) and len(idx) == 2 
            else idx 
            for idx in dual_edges_gdf.index
        ]
        dual_edges_gdf.index = pd.Index(new_edge_index)
    
    return dual_nodes_gdf, dual_edges_gdf


def _amenities_from_cached_gdf(features_obj):
    """The ``_c2g_gdf_pickle`` cache: no network, geometry intact. None if unusable."""
    cached = features_obj.get("_c2g_gdf_pickle")
    if not cached:
        return None

    import pickle
    import base64

    try:
        gdf = pickle.loads(base64.b64decode(cached))
    except Exception as e:
        print(f"Warning: Could not decode cached GeoDataFrame: {e}")
        return None

    if gdf is None or len(gdf) == 0:
        return None

    if gdf.crs is None:
        gdf = gdf.set_crs(features_obj.get("_c2g_gdf_crs", "EPSG:4326"))

    return gdf


def _amenities_from_place(features_obj):
    """Re-download amenity features for the object's ``place_name``. Only works for
    OSMnx objects geocoded from a place name; a point or bbox query leaves a
    place_name this cannot geocode, so those return None."""
    place = features_obj.get("place_name", "")

    if not place or place.startswith("Point") or place.startswith("BBox"):
        return None

    import osmnx as ox

    tags = {
        'amenity': ['cafe', 'restaurant', 'pub', 'bar', 'museum', 'theatre', 'cinema']
    }

    try:
        return ox.features_from_place(place, tags=tags)
    except Exception as e:
        print(f"Warning: Could not re-download amenities from '{place}': {e}")
        return None


def _amenities_from_mesh(features_obj):
    """Read vertex positions off the Blender mesh as a last resort."""
    if not features_obj.data or not getattr(features_obj.data, "vertices", None):
        return None

    import geopandas as gpd
    from shapely.geometry import Point

    points = [Point(v.co.x, v.co.y) for v in features_obj.data.vertices]

    if not points:
        return None

    original_crs = features_obj.get("crs", "EPSG:4326")
    return gpd.GeoDataFrame(geometry=points, crs=original_crs)


def prepare_amenities_from_features(features_obj, target_crs, limit=None):
    """Convert a features object to Point-geometry amenities in ``target_crs``,
    which must match the dual graph they get bridged to. Tries the pickled cache,
    then a re-download by place name, then raw mesh vertices; ValueError if none."""
    import geopandas as gpd

    amenities_gdf = _amenities_from_cached_gdf(features_obj)

    if amenities_gdf is None or len(amenities_gdf) == 0:
        amenities_gdf = _amenities_from_place(features_obj)

    if amenities_gdf is None or len(amenities_gdf) == 0:
        amenities_gdf = _amenities_from_mesh(features_obj)

    if amenities_gdf is None or len(amenities_gdf) == 0:
        raise ValueError("Could not extract amenities from the selected object")

    amenities_gdf = amenities_gdf.reset_index(drop=True)
    amenities_gdf = amenities_gdf.to_crs(target_crs)
    amenities_gdf['geometry'] = amenities_gdf.geometry.centroid
    amenities_gdf = gpd.GeoDataFrame(amenities_gdf[['geometry']], crs=amenities_gdf.crs)

    if limit:
        amenities_gdf = amenities_gdf.head(limit)

    amenities_gdf = amenities_gdf.reset_index(drop=True)

    return amenities_gdf


def bridge_amenities_to_segments(amenities_gdf, dual_nodes_gdf, k=1):
    """Connect each amenity to its ``k`` nearest street segments, returning the
    ``(nodes_dict, edges_dict)`` pair a heterogeneous graph takes. Both input
    frames gain a ``node_type`` column in place if they lack one."""
    import city2graph as c2g

    if 'node_type' not in amenities_gdf.columns:
        amenities_gdf['node_type'] = 'amenity'
    if 'node_type' not in dual_nodes_gdf.columns:
        dual_nodes_gdf['node_type'] = 'segment'

    nodes_dict = {
        "amenity": amenities_gdf,
        "segment": dual_nodes_gdf
    }

    # Only the bridge edges land here; the segment-to-segment edges of the dual
    # graph are not passed in.
    edges_dict = {}

    _, bridged_edges = c2g.bridge_nodes(
        nodes_dict=nodes_dict,
        proximity_method="knn",
        source_node_types=["amenity"],
        target_node_types=["segment"],
        k=k
    )
    
    edges_dict.update(bridged_edges)
    
    return nodes_dict, edges_dict


def compute_metapaths(nodes_dict, edges_dict, hops=3, directed=False):
    """Add amenity-to-amenity metapath edges across ``hops`` street segments: the
    sequence runs amenity to segment, ``hops`` segment-to-segment steps, then
    segment back to amenity."""
    import city2graph as c2g
    
    sequence = [("amenity", "is_nearby", "segment")]
    for _ in range(hops):
        sequence.append(("segment", "connects_to", "segment"))
    sequence.append(("segment", "is_nearby", "amenity"))
    
    result_nodes, result_edges = c2g.add_metapaths(
        graph=(nodes_dict, edges_dict),
        sequence=sequence,
        edge_attr=None,
        edge_attr_agg="sum",
        directed=directed,
        trace_path=False,
        as_nx=False,
        multigraph=True
    )
    
    return result_nodes, result_edges


def compute_metapaths_by_weight(nodes_dict, edges_dict, weight_attr, threshold, 
                                 endpoint_type="amenity", min_threshold=0.0,
                                 directed=False, new_relation_name=None):
    """Connect ``endpoint_type`` nodes reachable within a cost band, by Dijkstra and
    needing city2graph 0.3.1 or newer. Cost comes from the ``weight_attr`` edge
    attribute, and a pair connects when it falls between the two thresholds."""
    import city2graph as c2g
    
    result_nodes, result_edges = c2g.add_metapaths_by_weight(
        graph=(nodes_dict, edges_dict),
        weight=weight_attr,
        threshold=threshold,
        min_threshold=min_threshold,
        endpoint_type=endpoint_type,
        new_relation_name=new_relation_name,
        directed=directed,
        as_nx=False,
        multigraph=False
    )
    
    return result_nodes, result_edges


def extract_metapath_connections(result_edges, metapath_key=('amenity', 'metapath_0', 'amenity'), 
                                 add_multiplicity=True):
    """Pull the metapath edges out of an ``add_metapaths`` result, falling back to
    the first key whose relation name contains 'metapath'. ``add_multiplicity``
    collapses paths sharing endpoints into one row carrying how many there were,
    matched on coordinates rounded to 6 decimals and sorted, so direction is free."""
    import geopandas as gpd
    import pandas as pd
    
    metapath_gdf = None
    if metapath_key in result_edges:
        metapath_gdf = result_edges[metapath_key]
    else:
        for key in result_edges.keys():
            if 'metapath' in key[1]:
                metapath_gdf = result_edges[key]
                break
    
    if metapath_gdf is None or len(metapath_gdf) == 0:
        return None
    
    if not add_multiplicity:
        return metapath_gdf
    
    grouped_metapaths = []

    for idx, row in metapath_gdf.iterrows():
        if hasattr(row, 'geometry') and row.geometry:
            coords = list(row.geometry.coords)
            if len(coords) >= 2:
                start = (round(coords[0][0], 6), round(coords[0][1], 6))
                end = (round(coords[-1][0], 6), round(coords[-1][1], 6))
                # Sorted, so the key is the same in either direction.
                key = tuple(sorted([start, end]))
                
                grouped_metapaths.append({
                    'geometry': row.geometry,
                    'key': key,
                    'index': idx
                })
    
    from collections import defaultdict
    key_to_data = defaultdict(list)

    for item in grouped_metapaths:
        key_to_data[item['key']].append(item)

    unique_rows = []
    for key, items in key_to_data.items():
        first_item = items[0]
        multiplicity = len(items)

        original_row = metapath_gdf.loc[first_item['index']].copy()

        row_dict = {
            'geometry': first_item['geometry'],
            'multiplicity': multiplicity
        }

        for col in metapath_gdf.columns:
            if col != 'geometry':
                row_dict[col] = original_row[col]

        unique_rows.append(row_dict)

    result_gdf = gpd.GeoDataFrame(unique_rows, crs=metapath_gdf.crs)
    
    print(f"Metapaths grouped: {len(metapath_gdf)} raw → {len(result_gdf)} unique with multiplicity")
    
    return result_gdf

