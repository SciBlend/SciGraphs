# Export utilities for graph data

import os
import json
import xml.etree.ElementTree as ET


def export_to_graphml(graph_data, filepath, node_attributes=None):
    """
    Export graph to GraphML format.
    
    Args:
        graph_data: GraphData object
        filepath: Output file path
        node_attributes: Dict mapping attribute names to value arrays
    
    Returns:
        bool: Success status
    """
    try:
        # Create XML structure
        graphml = ET.Element('graphml')
        graphml.set('xmlns', 'http://graphml.graphdrawing.org/xmlns')
        
        graph = ET.SubElement(graphml, 'graph')
        graph.set('id', 'G')
        graph.set('edgedefault', 'directed' if hasattr(graph_data, 'is_directed') and graph_data.is_directed else 'undirected')
        
        # Define attribute keys
        if node_attributes:
            for attr_name in node_attributes.keys():
                key = ET.SubElement(graphml, 'key')
                key.set('id', attr_name)
                key.set('for', 'node')
                key.set('attr.name', attr_name)
                key.set('attr.type', 'double')
        
        # Add nodes
        for i, node in enumerate(graph_data.nodes):
            node_elem = ET.SubElement(graph, 'node')
            node_elem.set('id', str(node))
            
            # Add attributes
            if node_attributes:
                for attr_name, values in node_attributes.items():
                    data = ET.SubElement(node_elem, 'data')
                    data.set('key', attr_name)
                    data.text = str(values[i])
        
        # Add edges
        for i, edge in enumerate(graph_data.edges):
            edge_elem = ET.SubElement(graph, 'edge')
            edge_elem.set('source', str(edge[0]))
            edge_elem.set('target', str(edge[1]))
        
        # Write to file
        tree = ET.ElementTree(graphml)
        ET.indent(tree, space="  ")
        tree.write(filepath, encoding='utf-8', xml_declaration=True)
        
        return True
    except Exception as e:
        print(f"GraphML export error: {e}")
        return False


def export_to_gexf(graph_data, filepath, node_attributes=None):
    """
    Export graph to GEXF format (Gephi).
    
    Args:
        graph_data: GraphData object
        filepath: Output file path
        node_attributes: Dict mapping attribute names to value arrays
    
    Returns:
        bool: Success status
    """
    try:
        # Create XML structure
        gexf = ET.Element('gexf')
        gexf.set('xmlns', 'http://www.gexf.net/1.2draft')
        gexf.set('version', '1.2')
        
        graph = ET.SubElement(gexf, 'graph')
        graph.set('mode', 'static')
        graph.set('defaultedgetype', 'directed' if hasattr(graph_data, 'is_directed') and graph_data.is_directed else 'undirected')
        
        # Define attributes
        if node_attributes:
            attributes = ET.SubElement(graph, 'attributes')
            attributes.set('class', 'node')
            
            for attr_name in node_attributes.keys():
                attribute = ET.SubElement(attributes, 'attribute')
                attribute.set('id', attr_name)
                attribute.set('title', attr_name)
                attribute.set('type', 'double')
        
        # Add nodes
        nodes = ET.SubElement(graph, 'nodes')
        for i, node in enumerate(graph_data.nodes):
            node_elem = ET.SubElement(nodes, 'node')
            node_elem.set('id', str(node))
            node_elem.set('label', str(node))
            
            # Add attributes
            if node_attributes:
                attvalues = ET.SubElement(node_elem, 'attvalues')
                for attr_name, values in node_attributes.items():
                    attvalue = ET.SubElement(attvalues, 'attvalue')
                    attvalue.set('for', attr_name)
                    attvalue.set('value', str(values[i]))
        
        # Add edges
        edges = ET.SubElement(graph, 'edges')
        for i, edge in enumerate(graph_data.edges):
            edge_elem = ET.SubElement(edges, 'edge')
            edge_elem.set('id', str(i))
            edge_elem.set('source', str(edge[0]))
            edge_elem.set('target', str(edge[1]))
        
        # Write to file
        tree = ET.ElementTree(gexf)
        ET.indent(tree, space="  ")
        tree.write(filepath, encoding='utf-8', xml_declaration=True)
        
        return True
    except Exception as e:
        print(f"GEXF export error: {e}")
        return False


def export_to_json(graph_data, filepath, node_attributes=None):
    """
    Export graph to JSON format.
    
    Args:
        graph_data: GraphData object
        filepath: Output file path
        node_attributes: Dict mapping attribute names to value arrays
    
    Returns:
        bool: Success status
    """
    try:
        data = {
            'nodes': [],
            'edges': [],
            'directed': hasattr(graph_data, 'is_directed') and graph_data.is_directed
        }
        
        # Add nodes
        for i, node in enumerate(graph_data.nodes):
            node_data = {'id': str(node)}
            
            if node_attributes:
                node_data['attributes'] = {}
                for attr_name, values in node_attributes.items():
                    node_data['attributes'][attr_name] = float(values[i])
            
            data['nodes'].append(node_data)
        
        # Add edges
        for edge in graph_data.edges:
            data['edges'].append({
                'source': str(edge[0]),
                'target': str(edge[1])
            })
        
        # Write to file
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return True
    except Exception as e:
        print(f"JSON export error: {e}")
        return False


def export_to_csv_edges(graph_data, filepath):
    """
    Export graph edges to CSV format.
    
    Args:
        graph_data: GraphData object
        filepath: Output file path
    
    Returns:
        bool: Success status
    """
    try:
        with open(filepath, 'w') as f:
            # Header
            f.write("source,target\n")
            
            # Edges
            for edge in graph_data.edges:
                f.write(f"{edge[0]},{edge[1]}\n")
        
        return True
    except Exception as e:
        print(f"CSV export error: {e}")
        return False


def export_to_pajek(graph_data, filepath):
    """
    Export graph to Pajek NET format.
    
    Args:
        graph_data: GraphData object
        filepath: Output file path
    
    Returns:
        bool: Success status
    """
    try:
        with open(filepath, 'w') as f:
            # Write vertices section
            f.write(f"*Vertices {len(graph_data.nodes)}\n")
            
            # Create node index mapping
            node_to_idx = {node: i+1 for i, node in enumerate(graph_data.nodes)}
            
            # Write node list with labels
            for i, node in enumerate(graph_data.nodes):
                f.write(f'{i+1} "{node}"\n')
            
            # Write edges section
            if hasattr(graph_data, 'is_directed') and graph_data.is_directed:
                f.write("*Arcs\n")
            else:
                f.write("*Edges\n")
            
            # Write edges
            for edge in graph_data.edges:
                source_idx = node_to_idx[edge[0]]
                target_idx = node_to_idx[edge[1]]
                f.write(f"{source_idx} {target_idx}\n")
        
        return True
    except Exception as e:
        print(f"Pajek export error: {e}")
        return False


def export_positions(obj, filepath):
    """
    Export node positions from Blender object to CSV.
    
    Args:
        obj: Blender object with graph data
        filepath: Output file path
    
    Returns:
        bool: Success status
    """
    try:
        if "node_positions" not in obj:
            return False
        
        positions = obj["node_positions"]
        num_nodes = obj.get("num_nodes", 0)
        
        with open(filepath, 'w') as f:
            f.write("node_id,x,y,z\n")
            
            for i in range(num_nodes):
                x = positions[i*3]
                y = positions[i*3 + 1]
                z = positions[i*3 + 2]
                f.write(f"{i},{x},{y},{z}\n")
        
        return True
    except Exception as e:
        print(f"Position export error: {e}")
        return False


def export_statistics_report(stats, filepath):
    """
    Export graph statistics to text report.
    
    Args:
        stats: Dictionary of statistics
        filepath: Output file path
    
    Returns:
        bool: Success status
    """
    try:
        with open(filepath, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("SCIGRAPHS - Graph Statistics Report\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Number of Nodes: {stats.get('num_nodes', 'N/A')}\n")
            f.write(f"Number of Edges: {stats.get('num_edges', 'N/A')}\n")
            f.write(f"Density: {stats.get('density', 'N/A'):.6f}\n")
            f.write(f"Global Clustering Coefficient: {stats.get('global_clustering', 'N/A'):.6f}\n")
            f.write(f"Average Path Length: {stats.get('average_path_length', 'N/A'):.6f}\n")
            f.write(f"Diameter: {stats.get('diameter', 'N/A')}\n")
            f.write(f"Assortativity: {stats.get('assortativity', 'N/A'):.6f}\n")
            
            if 'degree_distribution' in stats:
                deg_dist = stats['degree_distribution']
                f.write(f"\nDegree Distribution:\n")
                f.write(f"  Mean: {deg_dist.get('mean', 'N/A'):.3f}\n")
                f.write(f"  Median: {deg_dist.get('median', 'N/A'):.3f}\n")
                f.write(f"  Std Dev: {deg_dist.get('std', 'N/A'):.3f}\n")
                f.write(f"  Min: {deg_dist.get('min', 'N/A')}\n")
                f.write(f"  Max: {deg_dist.get('max', 'N/A')}\n")
            
            f.write("\n" + "=" * 60 + "\n")
        
        return True
    except Exception as e:
        print(f"Report export error: {e}")
        return False

