# Turns SQL query results into the internal GraphData format.

import numpy as np
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
    PANDAS_REASON = None
except ImportError:
    pd = None
    PANDAS_AVAILABLE = False
    PANDAS_REASON = ("pandas is not installed; install the 'sql' extra to "
                     "enable tabular import and the database backends")
from ..algorithms import graph
from . import db_connector
from scigraphs_core.logger import log


def load_graph_from_sql(profile, sql_query, source_col, target_col):
    """Build a graph from two columns of a SELECT; source_col/target_col are indices."""
    if not sql_query or not sql_query.strip():
        log("Error: No SQL query provided")
        return None
    
    log(f"Loading graph from SQL: {profile.name}")
    import time
    start_time = time.time()
    
    df, error = db_connector.execute_query(profile, sql_query)
    
    if error:
        log(f"Error executing query: {error}")
        return None
    
    if df is None or len(df) == 0:
        log("Query returned no results")
        return None
    
    log(f"  Query executed in {time.time() - start_time:.2f}s, {len(df)} rows")
    
    try:
        source_col_name = df.columns[source_col]
        target_col_name = df.columns[target_col]
    except IndexError:
        log(f"Error: Column index out of range. Query has {len(df.columns)} columns.")
        return None
    
    edges_start = time.time()
    source_values = df[source_col_name].values
    target_values = df[target_col_name].values
    edges = list(zip(source_values, target_values))
    log(f"  Edges extracted in {time.time() - edges_start:.2f}s")
    
    nodes_start = time.time()
    all_nodes = np.concatenate([source_values, target_values])
    
    try:
        nodes = np.unique(all_nodes)
    except TypeError:
        # np.unique cannot order mixed types; pandas can.
        nodes = pd.Series(all_nodes).dropna().unique()
    
    log(f"  Nodes extracted in {time.time() - nodes_start:.2f}s")
    
    graph_data = graph.GraphData(nodes, edges, df)
    
    log(f"Total load time: {time.time() - start_time:.2f}s")
    log(f"  Nodes: {len(nodes):,}, Edges: {len(edges):,}")
    
    return graph_data


def get_columns_from_query(profile, sql_query):
    """A query's column names, from wrapping it in a one-row limit. [] on failure."""
    if not sql_query or not sql_query.strip():
        return []
    
    sql_stripped = sql_query.strip().rstrip(';')
    
    if profile.db_type == 'SQLSERVER':
        # SQL Server has no LIMIT; TOP goes right after SELECT.
        if sql_stripped.upper().startswith('SELECT '):
            wrapped_query = sql_stripped.replace('SELECT ', 'SELECT TOP 1 ', 1)
        else:
            wrapped_query = sql_stripped
    else:
        wrapped_query = f"SELECT * FROM ({sql_stripped}) AS subq LIMIT 1"
    
    df, error = db_connector.execute_query(profile, wrapped_query)
    
    if error:
        # Some queries will not survive being wrapped in a subquery.
        wrapped_query = f"{sql_stripped} LIMIT 1"
        df, error = db_connector.execute_query(profile, wrapped_query)
    
    if error or df is None:
        log(f"Error getting columns: {error}")
        return []
    
    return list(df.columns)


def load_geospatial_graph_from_sql(
    profile,
    sql_query,
    source_col,
    target_col,
    lat_col=None,
    lon_col=None,
    weight_col=None
):
    """Load a graph whose nodes carry latitude and longitude. Every *_col argument
    is a column index into the result, and the returned GraphData gains
    node_coordinates keyed by str(node), plus edge_weights."""
    if not sql_query or not sql_query.strip():
        return None
    
    log(f"Loading geospatial graph from SQL: {profile.name}")
    import time
    start_time = time.time()
    
    df, error = db_connector.execute_query(profile, sql_query)
    
    if error:
        log(f"Error executing query: {error}")
        return None
    
    if df is None or len(df) == 0:
        log("Query returned no results")
        return None
    
    try:
        source_col_name = df.columns[source_col]
        target_col_name = df.columns[target_col]
        weight_col_name = df.columns[weight_col] if weight_col is not None else None
        lat_col_name = df.columns[lat_col] if lat_col is not None else None
        lon_col_name = df.columns[lon_col] if lon_col is not None else None
    except IndexError:
        log(f"Error: Column index out of range. Query has {len(df.columns)} columns.")
        return None
    
    source_values = df[source_col_name].values
    target_values = df[target_col_name].values
    edges = list(zip(source_values, target_values))
    
    all_nodes = np.concatenate([source_values, target_values])
    nodes = np.unique(all_nodes)
    
    node_coordinates = {}
    
    if lat_col_name and lon_col_name:
        log("  Extracting geospatial coordinates...")
        
        for node in nodes:
            # First row mentioning this node, on either end.
            mask = (df[source_col_name] == node) | (df[target_col_name] == node)
            if mask.any():
                row = df[mask].iloc[0]
                lat = row[lat_col_name]
                lon = row[lon_col_name]
                
                if pd.notna(lat) and pd.notna(lon):
                    node_coordinates[str(node)] = (float(lat), float(lon))
    
    edge_weights = None
    if weight_col_name:
        edge_weights = pd.to_numeric(df[weight_col_name], errors='coerce').fillna(0).values
    
    graph_data = graph.GraphData(nodes, edges, df)
    graph_data.node_coordinates = node_coordinates
    graph_data.edge_weights = edge_weights
    graph_data.source_column_name = source_col_name
    graph_data.target_column_name = target_col_name
    
    log(f"Total load time: {time.time() - start_time:.2f}s")
    log(f"  Nodes: {len(nodes):,}, Edges: {len(edges):,}")
    log(f"  Nodes with coordinates: {len(node_coordinates)}")
    
    return graph_data


def preview_query(profile, sql_query, max_rows=10):
    """(DataFrame, total_rows, error); total_rows falls back to the preview length."""
    if not sql_query or not sql_query.strip():
        return None, 0, "No query provided"
    
    sql_stripped = sql_query.strip().rstrip(';')
    
    if profile.db_type == 'SQLSERVER':
        if sql_stripped.upper().startswith('SELECT '):
            preview_query_sql = sql_stripped.replace(
                'SELECT ', f'SELECT TOP {max_rows} ', 1
            )
        else:
            preview_query_sql = sql_stripped
    else:
        preview_query_sql = f"{sql_stripped} LIMIT {max_rows}"
    
    df, error = db_connector.execute_query(profile, preview_query_sql)
    
    if error:
        return None, 0, error
    
    if df is None:
        return None, 0, "Query returned no results"
    
    total_rows = len(df)
    try:
        count_query = f"SELECT COUNT(*) FROM ({sql_stripped}) AS count_subq"
        count_df, count_error = db_connector.execute_query(profile, count_query)
        if count_df is not None and len(count_df) > 0:
            total_rows = int(count_df.iloc[0, 0])
    except:
        pass
    
    return df, total_rows, None

