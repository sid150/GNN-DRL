"""
Core NetworkTopology implementation.
"""

import networkx as nx
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class NodeAttributes:
    """Data class for storing node attributes."""
    node_id: str
    processing_delay: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'node_id': self.node_id,
            'processing_delay': self.processing_delay
        }

    @staticmethod
    def validate(processing_delay: float) -> None:
        """Validate node attributes."""
        if processing_delay < 0:
            raise ValueError("processing_delay must be non-negative")


@dataclass
class LinkAttributes:
    """Data class for storing edge/link attributes."""
    capacity: float
    propagation_delay: float
    queue_size: int
    loss_probability: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            'capacity': self.capacity,
            'propagation_delay': self.propagation_delay,
            'queue_size': self.queue_size,
            'loss_probability': self.loss_probability
        }

    @staticmethod
    def validate(capacity: float, propagation_delay: float,
                 queue_size: int, loss_probability: float) -> None:
        """Validate link attributes."""
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        if propagation_delay < 0:
            raise ValueError("propagation_delay must be non-negative")
        if queue_size < 0:
            raise ValueError("queue_size must be non-negative")
        if not (0 <= loss_probability <= 1):
            raise ValueError("loss_probability must be between 0 and 1")


class NetworkTopology:
    """
    Represents a network graph with routers/switches (nodes) and links (edges).
    """

    def __init__(self):
        """Initialize the NetworkTopology with an empty directed graph."""
        self.graph = nx.DiGraph()
        self.node_attributes: Dict[str, Dict[str, Any]] = {}
        self.link_attributes: Dict[Tuple[str, str], Dict[str, Any]] = {}
        logger.info("NetworkTopology initialized")

    def add_node(self, node_id: str, processing_delay: float) -> None:
        """
        Add a node (router/switch) to the topology.

        Args:
            node_id (str): Unique identifier for the node
            processing_delay (float): Processing delay in milliseconds (must be >= 0)

        Raises:
            ValueError: If node_id is empty or processing_delay is invalid
            RuntimeError: If node already exists
        """
        if not node_id or not isinstance(node_id, str):
            raise ValueError("node_id must be a non-empty string")

        if node_id in self.node_attributes:
            raise RuntimeError(f"Node '{node_id}' already exists in topology")

        try:
            NodeAttributes.validate(processing_delay)
        except ValueError as e:
            raise ValueError(f"Invalid node attributes: {str(e)}")

        self.graph.add_node(node_id)
        node_attrs = NodeAttributes(node_id=node_id, processing_delay=processing_delay)
        self.node_attributes[node_id] = node_attrs.to_dict()

        logger.info(f"Added node '{node_id}' with processing_delay={processing_delay}ms")

    def add_link(self, source_node: str, dest_node: str, capacity: float,
                 propagation_delay: float, queue_size: int,
                 loss_probability: float = 0.0) -> None:
        """
        Add a directed link (edge) between two nodes.

        Args:
            source_node (str): Source node identifier
            dest_node (str): Destination node identifier
            capacity (float): Link capacity in Mbps (must be > 0)
            propagation_delay (float): Propagation delay in milliseconds (must be >= 0)
            queue_size (int): Queue size in packets (must be >= 0)
            loss_probability (float): Packet loss probability (0 to 1, default=0.0)

        Raises:
            ValueError: If either node doesn't exist or attributes are invalid
            RuntimeError: If link already exists
        """
        if source_node not in self.node_attributes:
            raise ValueError(f"Source node '{source_node}' not found in topology")
        if dest_node not in self.node_attributes:
            raise ValueError(f"Destination node '{dest_node}' not found in topology")

        if self.graph.has_edge(source_node, dest_node):
            raise RuntimeError(
                f"Link from '{source_node}' to '{dest_node}' already exists"
            )

        try:
            LinkAttributes.validate(capacity, propagation_delay, queue_size, loss_probability)
        except ValueError as e:
            raise ValueError(f"Invalid link attributes: {str(e)}")

        self.graph.add_edge(source_node, dest_node)
        link_key = (source_node, dest_node)
        link_attrs = LinkAttributes(
            capacity=capacity,
            propagation_delay=propagation_delay,
            queue_size=queue_size,
            loss_probability=loss_probability
        )
        self.link_attributes[link_key] = link_attrs.to_dict()

        logger.info(
            f"Added link '{source_node}' -> '{dest_node}' "
            f"(capacity={capacity}Mbps, delay={propagation_delay}ms, "
            f"queue={queue_size}, loss={loss_probability})"
        )

    def get_neighbors(self, node_id: str, direction: str = 'out') -> List[str]:
        """
        Get neighbors of a node.

        Args:
            node_id (str): Node identifier
            direction (str): Direction of neighbors ('out', 'in', 'all')

        Returns:
            List[str]: List of neighbor node identifiers
        """
        if node_id not in self.node_attributes:
            raise ValueError(f"Node '{node_id}' not found in topology")

        if direction not in ['out', 'in', 'all']:
            raise ValueError("direction must be 'out', 'in', or 'all'")

        neighbors = []
        try:
            if direction in ['out', 'all']:
                neighbors.extend(list(self.graph.successors(node_id)))
            if direction in ['in', 'all']:
                neighbors.extend(list(self.graph.predecessors(node_id)))
        except nx.NodeNotFound:
            raise ValueError(f"Node '{node_id}' not found in graph")

        return neighbors

    def get_shortest_path(self, source_node: str, dest_node: str,
                         weight: Optional[str] = None) -> List[str]:
        """
        Find the shortest path between two nodes.

        Args:
            source_node (str): Source node identifier
            dest_node (str): Destination node identifier
            weight (str, optional): Edge attribute to use as weight
                                   ('propagation_delay', 'capacity')

        Returns:
            List[str]: List of nodes representing the shortest path
        """
        if source_node not in self.node_attributes:
            raise ValueError(f"Source node '{source_node}' not found in topology")
        if dest_node not in self.node_attributes:
            raise ValueError(f"Destination node '{dest_node}' not found in topology")

        if source_node == dest_node:
            return [source_node]

        try:
            if weight is None:
                path = nx.shortest_path(self.graph, source_node, dest_node)
            else:
                if weight not in ['propagation_delay', 'capacity']:
                    raise ValueError(
                        f"weight must be 'propagation_delay' or 'capacity', got '{weight}'"
                    )

                if weight == 'capacity':
                    temp_graph = self.graph.copy()
                    for edge in temp_graph.edges():
                        link_key = (edge[0], edge[1])
                        if link_key in self.link_attributes:
                            capacity = self.link_attributes[link_key]['capacity']
                            temp_graph[edge[0]][edge[1]]['weight'] = 1.0 / capacity
                    path = nx.shortest_path(temp_graph, source_node, dest_node, weight='weight')
                else:
                    path = nx.shortest_path(
                        self.graph, source_node, dest_node, weight=weight
                    )

        except nx.NodeNotFound as e:
            raise ValueError(f"Node not found in graph: {str(e)}")
        except nx.NetworkXNoPath:
            raise nx.NetworkXNoPath(
                f"No path exists between '{source_node}' and '{dest_node}'"
            )

        logger.info(f"Found shortest path from '{source_node}' to '{dest_node}': {path}")
        return path

    def get_link_attributes(self, source_node: str, dest_node: str) -> Dict[str, Any]:
        """Get attributes of a specific link."""
        link_key = (source_node, dest_node)
        if link_key not in self.link_attributes:
            raise ValueError(
                f"Link from '{source_node}' to '{dest_node}' not found in topology"
            )
        return self.link_attributes[link_key].copy()

    def get_node_attributes(self, node_id: str) -> Dict[str, Any]:
        """Get attributes of a specific node."""
        if node_id not in self.node_attributes:
            raise ValueError(f"Node '{node_id}' not found in topology")
        return self.node_attributes[node_id].copy()

    def get_all_nodes(self) -> List[str]:
        """Get all node IDs in the topology."""
        return list(self.node_attributes.keys())

    def get_all_links(self) -> List[Tuple[str, str]]:
        """Get all link tuples in the topology."""
        return list(self.link_attributes.keys())

    def get_topology_stats(self) -> Dict[str, Any]:
        """Get statistics about the topology."""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_links': self.graph.number_of_edges(),
            'is_connected': nx.is_strongly_connected(self.graph),
            'density': nx.density(self.graph)
        }

    def __repr__(self) -> str:
        """String representation of the topology."""
        stats = self.get_topology_stats()
        return (
            f"NetworkTopology(nodes={stats['num_nodes']}, "
            f"links={stats['num_links']}, "
            f"connected={stats['is_connected']})"
        )
