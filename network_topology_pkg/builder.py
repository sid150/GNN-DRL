"""
Utility class for building common network topology patterns.
"""

from .core import NetworkTopology


class TopologyBuilder:
    """Utility class for building common network topology patterns."""

    @staticmethod
    def create_linear_topology(num_nodes: int, node_prefix: str = "R",
                              processing_delay: float = 2.0,
                              capacity: float = 100.0) -> NetworkTopology:
        """Create a linear chain topology."""
        topology = NetworkTopology()
        nodes = [f"{node_prefix}{i}" for i in range(num_nodes)]

        for node in nodes:
            topology.add_node(node, processing_delay=processing_delay)

        for i in range(len(nodes) - 1):
            topology.add_link(nodes[i], nodes[i + 1], capacity=capacity,
                            propagation_delay=1.0, queue_size=1000)

        return topology

    @staticmethod
    def create_mesh_topology(num_nodes: int, node_prefix: str = "R",
                            processing_delay: float = 2.0,
                            capacity: float = 100.0,
                            bidirectional: bool = True) -> NetworkTopology:
        """Create a mesh (fully connected) topology."""
        topology = NetworkTopology()
        nodes = [f"{node_prefix}{i}" for i in range(num_nodes)]

        for node in nodes:
            topology.add_node(node, processing_delay=processing_delay)

        for i, src in enumerate(nodes):
            for dst in nodes[i + 1:]:
                topology.add_link(src, dst, capacity=capacity,
                                propagation_delay=1.0, queue_size=1000)
                if bidirectional:
                    topology.add_link(dst, src, capacity=capacity,
                                    propagation_delay=1.0, queue_size=1000)

        return topology

    @staticmethod
    def create_ring_topology(num_nodes: int, node_prefix: str = "R",
                            processing_delay: float = 2.0,
                            capacity: float = 100.0) -> NetworkTopology:
        """Create a ring topology."""
        topology = NetworkTopology()
        nodes = [f"{node_prefix}{i}" for i in range(num_nodes)]

        for node in nodes:
            topology.add_node(node, processing_delay=processing_delay)

        for i in range(num_nodes):
            src = nodes[i]
            dst = nodes[(i + 1) % num_nodes]
            topology.add_link(src, dst, capacity=capacity,
                            propagation_delay=1.0, queue_size=1000)

        return topology

    @staticmethod
    def create_star_topology(num_edge_nodes: int = 5, node_prefix: str = "R",
                            core_processing_delay: float = 1.0,
                            edge_processing_delay: float = 3.0,
                            core_capacity: float = 1000.0,
                            edge_capacity: float = 100.0) -> NetworkTopology:
        """Create a star topology with a central core router."""
        topology = NetworkTopology()

        core_node = f"{node_prefix}_core"
        topology.add_node(core_node, processing_delay=core_processing_delay)

        for i in range(num_edge_nodes):
            edge_node = f"{node_prefix}_edge_{i}"
            topology.add_node(edge_node, processing_delay=edge_processing_delay)

            topology.add_link(core_node, edge_node, capacity=edge_capacity,
                            propagation_delay=5.0, queue_size=5000)
            topology.add_link(edge_node, core_node, capacity=edge_capacity,
                            propagation_delay=5.0, queue_size=5000)

        return topology

    @staticmethod
    def create_hierarchical_topology(num_core: int = 2,
                                   num_aggregation_per_core: int = 3,
                                   num_edge_per_aggregation: int = 4) -> NetworkTopology:
        """Create a hierarchical (fat-tree like) topology."""
        topology = NetworkTopology()

        core_nodes = [f"core_{i}" for i in range(num_core)]
        for node in core_nodes:
            topology.add_node(node, processing_delay=1.0)

        agg_nodes = []
        for i in range(num_core):
            for j in range(num_aggregation_per_core):
                agg_node = f"agg_{i}_{j}"
                agg_nodes.append(agg_node)
                topology.add_node(agg_node, processing_delay=2.0)

                for core_node in core_nodes:
                    topology.add_link(core_node, agg_node, capacity=500.0,
                                    propagation_delay=2.0, queue_size=5000)
                    topology.add_link(agg_node, core_node, capacity=500.0,
                                    propagation_delay=2.0, queue_size=5000)

        for i, agg_node in enumerate(agg_nodes):
            for j in range(num_edge_per_aggregation):
                edge_node = f"edge_{i}_{j}"
                topology.add_node(edge_node, processing_delay=3.0)

                topology.add_link(agg_node, edge_node, capacity=100.0,
                                propagation_delay=5.0, queue_size=2000)
                topology.add_link(edge_node, agg_node, capacity=100.0,
                                propagation_delay=5.0, queue_size=2000)

        return topology


    @staticmethod
    def create_random_topology(num_nodes: int,
                               num_edges: int,
                               node_prefix: str = "R",
                               processing_delay_range=(1.0, 5.0),
                               capacity_range=(50.0, 500.0),
                               delay_range=(1.0, 10.0)) -> NetworkTopology:
        """
        Create a random network topology.

        Args:
            num_nodes: number of nodes
            num_edges: number of directed edges to generate
            node_prefix: naming prefix for nodes
            processing_delay_range: (min, max) node processing delay
            capacity_range: (min, max) link capacity in Mbps
            delay_range: (min, max) propagation delay in ms

        Returns:
            NetworkTopology
        """
        import random
        topology = NetworkTopology()

        # ---- Add nodes ----
        nodes = [f"{node_prefix}{i}" for i in range(num_nodes)]
        for node in nodes:
            proc_delay = random.uniform(*processing_delay_range)
            topology.add_node(node, processing_delay=proc_delay)

        # ---- Generate all possible directed edges ----
        possible_edges = [
            (src, dst)
            for src in nodes
            for dst in nodes
            if src != dst
        ]

        # Ensure edges do not exceed possible combinations
        num_edges = min(num_edges, len(possible_edges))

        # ---- Randomly select edges ----
        chosen_edges = random.sample(possible_edges, num_edges)

        # ---- Add links ----
        for src, dst in chosen_edges:
            capacity = random.uniform(*capacity_range)
            prop_delay = random.uniform(*delay_range)

            topology.add_link(
                source_node=src,
                dest_node=dst,
                capacity=capacity,
                propagation_delay=prop_delay,
                queue_size=1000
            )

        return topology
