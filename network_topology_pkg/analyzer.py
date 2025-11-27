"""
Analysis tools for network topologies.
"""

from .core import NetworkTopology
from typing import List, Dict, Tuple
import networkx as nx


class TopologyAnalyzer:

    def __init__(self, topology: NetworkTopology):
        """Initialize analyzer with a topology instance."""
        self.topology = topology

    def get_total_path_delay(self, path: List[str]) -> float:
        """Calculate total propagation and processing delay along a path."""
        if len(path) < 2:
            raise ValueError("Path must contain at least 2 nodes")

        total_delay = 0.0

        for node in path[:-1]:
            node_attrs = self.topology.get_node_attributes(node)
            total_delay += node_attrs['processing_delay']

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            link_attrs = self.topology.get_link_attributes(src, dst)
            total_delay += link_attrs['propagation_delay']

        return total_delay

    def get_bottleneck_capacity(self, path: List[str]) -> float:
        """Get the minimum link capacity (bottleneck) along a path."""
        if len(path) < 2:
            raise ValueError("Path must contain at least 2 nodes")

        min_capacity = float('inf')

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            link_attrs = self.topology.get_link_attributes(src, dst)
            capacity = link_attrs['capacity']
            min_capacity = min(min_capacity, capacity)

        return min_capacity

    def get_path_reliability(self, path: List[str]) -> float:
        """Calculate end-to-end packet delivery probability along a path."""
        if len(path) < 2:
            raise ValueError("Path must contain at least 2 nodes")

        reliability = 1.0

        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            link_attrs = self.topology.get_link_attributes(src, dst)
            loss_prob = link_attrs['loss_probability']
            reliability *= (1.0 - loss_prob)

        return reliability

    def find_disjoint_paths(self, source: str, dest: str,
                           num_paths: int = 2) -> List[List[str]]:
        """Find edge-disjoint paths between two nodes."""
        paths = []
        temp_graph = self.topology.graph.copy()

        for _ in range(num_paths):
            try:
                path = nx.shortest_path(temp_graph, source, dest)
                paths.append(path)

                for i in range(len(path) - 1):
                    temp_graph.remove_edge(path[i], path[i + 1])
            except nx.NetworkXNoPath:
                break

        return paths

    def get_network_centrality(self) -> Dict[str, float]:
        """Calculate betweenness centrality for all nodes."""
        return dict(nx.betweenness_centrality(self.topology.graph))

    def get_heavily_used_links(self, threshold: float = 0.5) -> List[Tuple[str, str]]:
        """Identify links that might be heavily used (low capacity)."""
        bottleneck_links = []

        for src, dst in self.topology.get_all_links():
            link_attrs = self.topology.get_link_attributes(src, dst)
            if link_attrs['capacity'] < threshold:
                bottleneck_links.append((src, dst))

        return bottleneck_links

    def get_node_degree_distribution(self) -> Dict[int, int]:
        """Get the degree distribution of the network."""
        degree_count = {}
        for node in self.topology.graph.nodes():
            degree = self.topology.graph.degree(node)
            degree_count[degree] = degree_count.get(degree, 0) + 1
        return degree_count
