class GraphData:
    """A simple container for graph information."""
    def __init__(self, nodes, edges, dataframe=None):
        self.nodes = list(nodes)
        self.edges = list(edges)
        self.dataframe = dataframe
        
        self.node_to_index = {node: i for i, node in enumerate(self.nodes)}

