import json
from ...utils.logger import log
from . import utils


def export_to_graphml(obj, filepath):
    """
    Export Blender graph object to GraphML format.
    
    Args:
        obj: Blender mesh object representing a graph
        filepath: Output file path (.graphml)
    
    Returns:
        bool: True if export succeeded
    """
    try:
        import networkx as nx
        
        G = utils.extract_graph_from_blender(obj)
        if G is None:
            log("Failed to extract graph from object")
            return False
        
        log(f"Exporting graph to GraphML: {filepath}")
        
        for node, data in G.nodes(data=True):
            for key, value in list(data.items()):
                if isinstance(value, (list, tuple)):
                    data[key] = str(value)
        
        for u, v, data in G.edges(data=True):
            for key, value in list(data.items()):
                if isinstance(value, (list, tuple)):
                    data[key] = str(value)
        
        nx.write_graphml(G, filepath)
        log(f"Graph exported successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return True
        
    except Exception as e:
        log(f"Error exporting to GraphML: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_to_json(obj, filepath):
    """
    Export Blender graph object to JSON format (node-link data).
    
    Args:
        obj: Blender mesh object representing a graph
        filepath: Output file path (.json)
    
    Returns:
        bool: True if export succeeded
    """
    try:
        import networkx as nx
        from networkx.readwrite import json_graph
        
        G = utils.extract_graph_from_blender(obj)
        if G is None:
            log("Failed to extract graph from object")
            return False
        
        log(f"Exporting graph to JSON: {filepath}")
        
        data = json_graph.node_link_data(G)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        log(f"Graph exported successfully: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        return True
        
    except Exception as e:
        log(f"Error exporting to JSON: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_graph(obj, filepath, format='JSON'):
    """
    Export graph to specified format.
    
    Args:
        obj: Blender mesh object representing a graph
        filepath: Output file path
        format: Export format ('JSON' or 'GRAPHML')
    
    Returns:
        bool: True if export succeeded
    """
    format = format.upper()
    
    if format == 'JSON':
        return export_to_json(obj, filepath)
    elif format == 'GRAPHML':
        return export_to_graphml(obj, filepath)
    else:
        log(f"Unknown export format: {format}")
        return False


EXTERNAL_LOADER_SCRIPT = '''
"""
External PyTorch Geometric Loader for SciGraphs/City2Graph Exports

This script loads graph data exported from Blender SciGraphs and converts it
to PyTorch Geometric HeteroData format for GNN training.

Usage:
    python load_scigraphs_export.py exported_graph.json

Requirements:
    pip install torch torch_geometric networkx

Author: SciGraphs/City2Graph Integration
"""

import json
import torch
import networkx as nx
from torch_geometric.data import HeteroData


def load_json_export(filepath):
    """Load JSON node-link format from SciGraphs export."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return nx.node_link_graph(data)


def load_graphml_export(filepath):
    """Load GraphML format from SciGraphs export."""
    return nx.read_graphml(filepath)


def networkx_to_pyg_hetero(G):
    """
    Convert NetworkX graph to PyTorch Geometric HeteroData.
    
    Assumes nodes have 'node_type' attribute and edges have 'edge_type' attribute.
    If not present, treats as homogeneous graph.
    """
    data = HeteroData()
    
    node_types = set()
    edge_types = set()
    
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get('node_type', 'default')
        node_types.add(node_type)
    
    for u, v, attrs in G.edges(data=True):
        edge_type = attrs.get('edge_type', 'connects')
        edge_types.add(edge_type)
    
    node_type_to_idx = {nt: {} for nt in node_types}
    
    for node, attrs in G.nodes(data=True):
        node_type = attrs.get('node_type', 'default')
        idx = len(node_type_to_idx[node_type])
        node_type_to_idx[node_type][node] = idx
        
        if 'pos' in attrs:
            pos = attrs['pos']
            if isinstance(pos, str):
                pos = eval(pos)
            if node_type not in data.node_types:
                data[node_type].pos = []
            data[node_type].pos.append(pos)
    
    for node_type in node_types:
        if hasattr(data[node_type], 'pos'):
            data[node_type].pos = torch.tensor(data[node_type].pos, dtype=torch.float)
    
    for u, v, attrs in G.edges(data=True):
        u_type = G.nodes[u].get('node_type', 'default')
        v_type = G.nodes[v].get('node_type', 'default')
        edge_type = attrs.get('edge_type', 'connects')
        
        u_idx = node_type_to_idx[u_type][u]
        v_idx = node_type_to_idx[v_type][v]
        
        edge_key = (u_type, edge_type, v_type)
        
        if edge_key not in data.edge_types:
            data[edge_key].edge_index = []
        
        data[edge_key].edge_index.append([u_idx, v_idx])
    
    for edge_key in data.edge_types:
        edge_list = data[edge_key].edge_index
        data[edge_key].edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    
    return data


def load_scigraphs_export(filepath):
    """
    Main function to load SciGraphs export and convert to PyG HeteroData.
    
    Args:
        filepath: Path to .json or .graphml file
    
    Returns:
        HeteroData: PyTorch Geometric heterogeneous graph
    """
    if filepath.endswith('.json'):
        G = load_json_export(filepath)
    elif filepath.endswith('.graphml'):
        G = load_graphml_export(filepath)
    else:
        raise ValueError("File must be .json or .graphml")
    
    print(f"Loaded graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    data = networkx_to_pyg_hetero(G)
    
    print(f"Converted to HeteroData:")
    print(f"  Node types: {data.node_types}")
    print(f"  Edge types: {data.edge_types}")
    
    return data


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python load_scigraphs_export.py <exported_graph.json|.graphml>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    hetero_data = load_scigraphs_export(filepath)
    
    print("\\nExample: Using with PyG GNN model")
    print("  model = HeteroGNN(...)")
    print("  out = model(hetero_data.x_dict, hetero_data.edge_index_dict)")
'''


def get_external_loader_script():
    """
    Get the external PyTorch Geometric loader script.
    
    Returns:
        str: Python script for loading SciGraphs exports in external environments
    """
    return EXTERNAL_LOADER_SCRIPT


def save_external_loader_script(filepath):
    """
    Save the external loader script to a file.
    
    Args:
        filepath: Output path for the Python script
    
    Returns:
        bool: True if save succeeded
    """
    try:
        with open(filepath, 'w') as f:
            f.write(EXTERNAL_LOADER_SCRIPT)
        log(f"External loader script saved to: {filepath}")
        return True
    except Exception as e:
        log(f"Error saving external loader script: {e}")
        return False

