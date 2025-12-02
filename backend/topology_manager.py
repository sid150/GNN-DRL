"""
Network topology management module.
Handles topology building, validation, and manipulation.
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import json


@dataclass
class TopologyMetadata:
    """Metadata about a topology."""
    name: str
    num_nodes: int
    num_edges: int
    average_degree: float
    density: float
    diameter: int
    is_connected: bool


class NetworkTopologyBuilder:
    """Builder for network topologies with various generation methods."""
    
    def __init__(self):
        """Initialize topology builder."""
        self.current_topology = None
        self.current_graph = None
    
    def build_geant2(self, num_nodes: int = 40) -> nx.Graph:
        """Build GEANT2 network topology.
        
        Args:
            num_nodes: Number of nodes in GEANT2-like topology
        
        Returns:
            NetworkX graph
        """
        self.current_graph = nx.Graph()
        self.current_graph.add_nodes_from(range(num_nodes))
        
        # Add edges with realistic connectivity patterns
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),
            (5, 6), (6, 7), (7, 8), (8, 9), (9, 10),
            (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
            (15, 0), (2, 8), (4, 10), (6, 12), (8, 14),
            (0, 5), (2, 7), (4, 9), (6, 11), (8, 13)
        ]
        
        for src, dst in edges:
            if src < num_nodes and dst < num_nodes:
                self.current_graph.add_edge(src, dst)
        
        return self.current_graph
    
    def build_nsfnet(self, num_nodes: int = 14) -> nx.Graph:
        """Build NSFNet network topology.
        
        Args:
            num_nodes: Number of nodes (typically 14 for NSFNet)
        
        Returns:
            NetworkX graph
        """
        self.current_graph = nx.Graph()
        self.current_graph.add_nodes_from(range(num_nodes))
        
        # NSFNet topology edges
        edges = [
            (0, 1), (0, 2), (0, 7),
            (1, 2), (1, 3),
            (2, 5),
            (3, 4), (3, 6),
            (4, 5), (4, 12),
            (5, 6), (5, 8),
            (6, 7), (6, 13),
            (7, 8),
            (8, 9), (8, 11),
            (9, 10),
            (10, 11), (10, 12),
            (11, 12),
            (12, 13),
        ]
        
        for src, dst in edges:
            if src < num_nodes and dst < num_nodes:
                self.current_graph.add_edge(src, dst)
        
        return self.current_graph
    
    def build_random(self, num_nodes: int = 14, edge_probability: float = 0.3, 
                     seed: Optional[int] = None) -> nx.Graph:
        """Build random topology using Erdős-Rényi model.
        
        Args:
            num_nodes: Number of nodes
            edge_probability: Probability of edge between any two nodes
            seed: Random seed for reproducibility
        
        Returns:
            NetworkX graph
        """
        self.current_graph = nx.erdos_renyi_graph(num_nodes, edge_probability, seed=seed)
        
        # Ensure connected
        while not nx.is_connected(self.current_graph):
            self.current_graph = nx.erdos_renyi_graph(num_nodes, edge_probability, seed=seed)
        
        return self.current_graph
    
    def build_barabasi_albert(self, num_nodes: int = 14, m: int = 3,
                             seed: Optional[int] = None) -> nx.Graph:
        """Build scale-free topology using Barabási-Albert model.
        
        Args:
            num_nodes: Number of nodes
            m: Number of edges to attach from new node to existing nodes
            seed: Random seed for reproducibility
        
        Returns:
            NetworkX graph
        """
        self.current_graph = nx.barabasi_albert_graph(num_nodes, m, seed=seed)
        return self.current_graph
    
    def build_watts_strogatz(self, num_nodes: int = 14, k: int = 4,
                            p: float = 0.3, seed: Optional[int] = None) -> nx.Graph:
        """Build small-world topology using Watts-Strogatz model.
        
        Args:
            num_nodes: Number of nodes
            k: Each node is connected to k nearest neighbors
            p: Probability of rewiring edge
            seed: Random seed for reproducibility
        
        Returns:
            NetworkX graph
        """
        self.current_graph = nx.watts_strogatz_graph(num_nodes, k, p, seed=seed)
        return self.current_graph
    
    def add_link_weights(self, weight_type: str = "uniform",
                        capacity: float = 100.0) -> nx.Graph:
        """Add link weights (capacity, latency) to topology.
        
        Args:
            weight_type: Type of weights (uniform, random, geo-correlated)
            capacity: Link capacity in Mbps
        
        Returns:
            Modified graph with weights
        """
        if self.current_graph is None:
            raise ValueError("No topology built. Call a build_* method first.")
        
        for u, v in self.current_graph.edges():
            if weight_type == "uniform":
                self.current_graph[u][v]['capacity'] = capacity
                self.current_graph[u][v]['latency'] = 5.0  # ms
            elif weight_type == "random":
                self.current_graph[u][v]['capacity'] = capacity * np.random.uniform(0.5, 1.5)
                self.current_graph[u][v]['latency'] = 5.0 * np.random.uniform(0.5, 2.0)
            elif weight_type == "geo-correlated":
                # Simulate geographic distance correlation
                distance = np.random.exponential(scale=100)  # km
                self.current_graph[u][v]['latency'] = distance / 200 + 1  # Speed of light ~200km/ms
                self.current_graph[u][v]['capacity'] = capacity
        
        return self.current_graph
    
    def get_metadata(self) -> TopologyMetadata:
        """Get topology metadata.
        
        Returns:
            TopologyMetadata object
        """
        if self.current_graph is None:
            raise ValueError("No topology built")
        
        g = self.current_graph
        num_nodes = g.number_of_nodes()
        num_edges = g.number_edges()
        average_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
        density = nx.density(g)
        
        if nx.is_connected(g):
            diameter = nx.diameter(g)
        else:
            # Get largest connected component
            largest_cc = max(nx.connected_components(g), key=len)
            subgraph = g.subgraph(largest_cc)
            diameter = nx.diameter(subgraph)
        
        return TopologyMetadata(
            name=f"Network_{num_nodes}nodes",
            num_nodes=num_nodes,
            num_edges=num_edges,
            average_degree=average_degree,
            density=density,
            diameter=diameter,
            is_connected=nx.is_connected(g)
        )
    
    def to_adjacency_matrix(self) -> np.ndarray:
        """Convert current topology to adjacency matrix.
        
        Returns:
            Adjacency matrix as numpy array
        """
        if self.current_graph is None:
            raise ValueError("No topology built")
        
        return nx.to_numpy_array(self.current_graph)
    
    def to_edge_list(self) -> np.ndarray:
        """Convert current topology to edge list format.
        
        Returns:
            Edge list as (2, num_edges) numpy array
        """
        if self.current_graph is None:
            raise ValueError("No topology built")
        
        edges = np.array(list(self.current_graph.edges())).T
        return edges if edges.size > 0 else np.zeros((2, 0), dtype=int)
    
    def save_topology(self, filepath: str):
        """Save topology to file.
        
        Args:
            filepath: Path to save topology
        """
        if self.current_graph is None:
            raise ValueError("No topology built")
        
        nx.write_graphml(self.current_graph, filepath)
    
    def load_topology(self, filepath: str) -> nx.Graph:
        """Load topology from file.
        
        Args:
            filepath: Path to load topology from
        
        Returns:
            Loaded NetworkX graph
        """
        self.current_graph = nx.read_graphml(filepath)
        return self.current_graph


class NetworkTopologyValidator:
    """Validates network topologies for simulation suitability."""
    
    @staticmethod
    def validate(graph: nx.Graph, min_nodes: int = 5, max_nodes: int = 100,
                 require_connected: bool = True) -> Tuple[bool, List[str]]:
        """Validate topology.
        
        Args:
            graph: NetworkX graph to validate
            min_nodes: Minimum number of nodes
            max_nodes: Maximum number of nodes
            require_connected: Whether graph must be connected
        
        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []
        
        # Check node count
        num_nodes = graph.number_of_nodes()
        if num_nodes < min_nodes:
            errors.append(f"Too few nodes: {num_nodes} < {min_nodes}")
        if num_nodes > max_nodes:
            errors.append(f"Too many nodes: {num_nodes} > {max_nodes}")
        
        # Check connectivity
        if require_connected and not nx.is_connected(graph):
            errors.append("Graph is not connected")
        
        # Check for self-loops
        if nx.number_of_selfloops(graph) > 0:
            errors.append("Graph contains self-loops")
        
        # Check edge count (at least minimum for meaningful topology)
        min_edges = num_nodes - 1 if require_connected else 0
        if graph.number_of_edges() < min_edges:
            errors.append(f"Too few edges: {graph.number_of_edges()} < {min_edges}")
        
        return len(errors) == 0, errors
