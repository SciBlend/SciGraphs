# SQL Importer module for SciGraphs
#
# Handles importing graph data from SQL database queries.
# Converts query results to the internal GraphData format.

import numpy as np
import pandas as pd
from ..algorithms import graph
from . import db_connector
from ...utils.logger import log


def load_graph_from_sql(profile, sql_query, source_col, target_col):
    """
    Execute a SQL query and extract graph data from the results.
    
    Args:
        profile: DatabaseConnectionProfile instance
        sql_query: SQL query string (SELECT only)
        source_col: Index of the source column in query results
        target_col: Index of the target column in query results
        
    Returns:
        GraphData object or None if failed
    """
    if not sql_query or not sql_query.strip():
        log("Error: No SQL query provided")
        return None
    
    log(f"Loading graph from SQL: {profile.name}")
    import time
    start_time = time.time()
    
    # Execute the query
    df, error = db_connector.execute_query(profile, sql_query)
    
    if error:
        log(f"Error executing query: {error}")
        return None
    
    if df is None or len(df) == 0:
        log("Query returned no results")
        return None
    
    log(f"  Query executed in {time.time() - start_time:.2f}s, {len(df)} rows")
    
    # Get column names from indices
    try:
        source_col_name = df.columns[source_col]
        target_col_name = df.columns[target_col]
    except IndexError:
        log(f"Error: Column index out of range. Query has {len(df.columns)} columns.")
        return None
    
    # Extract edges
    edges_start = time.time()
    source_values = df[source_col_name].values
    target_values = df[target_col_name].values
    edges = list(zip(source_values, target_values))
    log(f"  Edges extracted in {time.time() - edges_start:.2f}s")
    
    # Get unique nodes
    nodes_start = time.time()
    all_nodes = np.concatenate([source_values, target_values])
    
    try:
        nodes = np.unique(all_nodes)
    except TypeError:
        # Handle mixed types by using pandas
        nodes = pd.Series(all_nodes).dropna().unique()
    
    log(f"  Nodes extracted in {time.time() - nodes_start:.2f}s")
    
    # Create graph data object
    graph_data = graph.GraphData(nodes, edges, df)
    
    log(f"Total load time: {time.time() - start_time:.2f}s")
    log(f"  Nodes: {len(nodes):,}, Edges: {len(edges):,}")
    
    return graph_data


def get_columns_from_query(profile, sql_query):
    """
    Execute a query with LIMIT 0 to get column names without fetching data.
    Falls back to LIMIT 1 if LIMIT 0 is not supported.
    
    Args:
        profile: DatabaseConnectionProfile instance
        sql_query: SQL query string
        
    Returns:
        List of column names or empty list if failed
    """
    if not sql_query or not sql_query.strip():
        return []
    
    # Wrap query with LIMIT to avoid fetching all data
    # Different databases have different LIMIT syntax
    sql_stripped = sql_query.strip().rstrip(';')
    
    if profile.db_type == 'SQLSERVER':
        # SQL Server uses TOP instead of LIMIT
        # Need to inject TOP after SELECT
        if sql_stripped.upper().startswith('SELECT '):
            wrapped_query = sql_stripped.replace('SELECT ', 'SELECT TOP 1 ', 1)
        else:
            wrapped_query = sql_stripped
    else:
        # PostgreSQL, MySQL, SQLite use LIMIT
        wrapped_query = f"SELECT * FROM ({sql_stripped}) AS subq LIMIT 1"
    
    df, error = db_connector.execute_query(profile, wrapped_query)
    
    if error:
        # Try original query with simple LIMIT appended
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
    """
    Load graph with geospatial coordinates from SQL query.
    
    Args:
        profile: DatabaseConnectionProfile instance
        sql_query: SQL query string
        source_col: Index of source column
        target_col: Index of target column
        lat_col: Index of latitude column (optional)
        lon_col: Index of longitude column (optional)
        weight_col: Index of weight column (optional)
        
    Returns:
        GraphData object with node_coordinates attribute
    """
    if not sql_query or not sql_query.strip():
        return None
    
    log(f"Loading geospatial graph from SQL: {profile.name}")
    import time
    start_time = time.time()
    
    # Execute the query
    df, error = db_connector.execute_query(profile, sql_query)
    
    if error:
        log(f"Error executing query: {error}")
        return None
    
    if df is None or len(df) == 0:
        log("Query returned no results")
        return None
    
    # Get column names
    try:
        source_col_name = df.columns[source_col]
        target_col_name = df.columns[target_col]
        weight_col_name = df.columns[weight_col] if weight_col is not None else None
        lat_col_name = df.columns[lat_col] if lat_col is not None else None
        lon_col_name = df.columns[lon_col] if lon_col is not None else None
    except IndexError:
        log(f"Error: Column index out of range. Query has {len(df.columns)} columns.")
        return None
    
    # Extract edges
    source_values = df[source_col_name].values
    target_values = df[target_col_name].values
    edges = list(zip(source_values, target_values))
    
    # Get unique nodes
    all_nodes = np.concatenate([source_values, target_values])
    nodes = np.unique(all_nodes)
    
    # Handle geospatial coordinates
    node_coordinates = {}
    
    if lat_col_name and lon_col_name:
        log("  Extracting geospatial coordinates...")
        
        for node in nodes:
            # Find first row with this node
            mask = (df[source_col_name] == node) | (df[target_col_name] == node)
            if mask.any():
                row = df[mask].iloc[0]
                lat = row[lat_col_name]
                lon = row[lon_col_name]
                
                if pd.notna(lat) and pd.notna(lon):
                    node_coordinates[str(node)] = (float(lat), float(lon))
    
    # Extract edge weights if provided
    edge_weights = None
    if weight_col_name:
        edge_weights = pd.to_numeric(df[weight_col_name], errors='coerce').fillna(0).values
    
    # Create graph data object
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
    """
    Execute a query and return a preview of the results.
    
    Args:
        profile: DatabaseConnectionProfile instance
        sql_query: SQL query string
        max_rows: Maximum number of rows to return
        
    Returns:
        Tuple of (DataFrame preview, total_rows, error_message)
    """
    if not sql_query or not sql_query.strip():
        return None, 0, "No query provided"
    
    # Wrap query with LIMIT for preview
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
    
    # Try to get total count (optional, may fail)
    total_rows = len(df)
    try:
        count_query = f"SELECT COUNT(*) FROM ({sql_stripped}) AS count_subq"
        count_df, count_error = db_connector.execute_query(profile, count_query)
        if count_df is not None and len(count_df) > 0:
            total_rows = int(count_df.iloc[0, 0])
    except:
        pass
    
    return df, total_rows, None

