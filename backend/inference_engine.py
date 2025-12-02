"""
Inference engine for routing decisions on new topologies.
Handles state preprocessing and action selection.
"""

import torch
import numpy as np
import networkx as nx
from typing import Dict, Tuple, List, Optional

# Handle both package and script imports
try:
    from .gnn_agent import GNNAgent, GCNLayer, create_gnn_agent
except ImportError:
    from gnn_agent import GNNAgent, GCNLayer, create_gnn_agent


class StatePreprocessor:
    """Preprocesses network state for GNN input."""
    
    def __init__(self, num_nodes: int):
        """Initialize state preprocessor.
        
        Args:
            num_nodes: Number of nodes in network
        """
        self.num_nodes = num_nodes
    
    def get_node_features(self, graph: nx.Graph, link_states: Dict) -> torch.Tensor:
        """Extract node features from graph and link states.
        
        Args:
            graph: Network topology graph
            link_states: Current link state information
        
        Returns:
            Node feature tensor (num_nodes, feature_dim)
        """
        features = []
        
        for node in range(self.num_nodes):
            # Node degree
            degree = graph.degree(node) if node in graph else 0
            
            # Incoming/outgoing link utilization
            in_util = 0.0
            out_util = 0.0
            in_links = 0
            out_links = 0
            
            for (u, v), state in link_states.items():
                if v == node:
                    in_util += state.utilization
                    in_links += 1
                if u == node:
                    out_util += state.utilization
                    out_links += 1
            
            in_util = in_util / max(in_links, 1)
            out_util = out_util / max(out_links, 1)
            
            # Normalize
            node_features = [
                degree / max(self.num_nodes - 1, 1),  # Normalized degree
                in_util / 100.0,  # Normalized input utilization
                out_util / 100.0,  # Normalized output utilization
            ]
            
            # Pad to fixed dimension (128)
            while len(node_features) < 128:
                node_features.append(0.0)
            
            features.append(node_features[:128])
        
        return torch.FloatTensor(features)
    
    def get_edge_features(self, graph: nx.Graph, link_states: Dict) -> torch.Tensor:
        """Extract edge features from graph.
        
        Args:
            graph: Network topology graph
            link_states: Current link state information
        
        Returns:
            Edge index tensor (2, num_edges)
        """
        edges = list(graph.edges())
        
        if len(edges) == 0:
            return torch.zeros((2, 0), dtype=torch.long)
        
        edge_index = np.array(edges).T
        return torch.LongTensor(edge_index)
    
    def get_adjacency_matrix(self, graph: nx.Graph) -> torch.Tensor:
        """Get adjacency matrix from graph.
        
        Args:
            graph: Network topology graph
        
        Returns:
            Adjacency matrix tensor
        """
        adj = nx.to_numpy_array(graph)
        return torch.FloatTensor(adj)
    
    def encode_flow_state(self, flow_id: int, source: int, destination: int) -> torch.Tensor:
        """Encode flow state for agent input.
        
        Args:
            flow_id: Flow identifier
            source: Source node
            destination: Destination node
        
        Returns:
            Encoded flow state tensor (1, 3)
        """
        flow_state = [
            flow_id / 1000.0,  # Normalize flow ID
            source / self.num_nodes,  # Normalize source
            destination / self.num_nodes  # Normalize destination
        ]
        return torch.FloatTensor([flow_state])


class GNNInferenceEngine:
    """Inference engine for GNN-based routing."""
    
    def __init__(self, model: Optional[GNNAgent] = None, device: str = 'cpu'):
        """Initialize inference engine.
        
        Args:
            model: GNN agent model (or creates new if None)
            device: Device to run inference on (cpu or cuda)
        """
        self.device = device
        self.model = model or create_gnn_agent()
        self.model.to(device)
        self.model.eval()
        
        self.preprocessor = None
    
    def setup(self, topology: nx.Graph):
        """Setup inference engine for a topology.
        
        Args:
            topology: Network topology graph
        """
        num_nodes = topology.number_of_nodes()
        self.preprocessor = StatePreprocessor(num_nodes)
    
    def infer_routing_action(self, graph: nx.Graph, link_states: Dict,
                            flow_id: int, source: int, destination: int,
                            deterministic: bool = True) -> int:
        """Infer routing action for a flow.
        
        Args:
            graph: Network topology
            link_states: Current link states
            flow_id: Flow identifier
            source: Source node
            destination: Destination node
            deterministic: Whether to select action deterministically
        
        Returns:
            Routing action (0=shortest, 1=min-delay, 2=max-capacity)
        """
        if self.preprocessor is None:
            self.setup(graph)
        
        # Preprocess state
        node_features = self.preprocessor.get_node_features(graph, link_states)
        edge_index = self.preprocessor.get_edge_features(graph, link_states)
        flow_state = self.preprocessor.encode_flow_state(flow_id, source, destination)
        adjacency = self.preprocessor.get_adjacency_matrix(graph)
        
        # Move to device
        node_features = node_features.to(self.device)
        edge_index = edge_index.to(self.device)
        flow_state = flow_state.to(self.device)
        adjacency = adjacency.to(self.device)
        
        # Inference
        with torch.no_grad():
            action = self.model.select_action(
                node_features, edge_index, flow_state, adjacency,
                deterministic=deterministic
            )
        
        return action
    
    def infer_batch(self, graph: nx.Graph, link_states: Dict,
                   flows: List[Tuple[int, int, int]],
                   deterministic: bool = True) -> List[int]:
        """Infer routing actions for multiple flows.
        
        Args:
            graph: Network topology
            link_states: Current link states
            flows: List of (flow_id, source, destination) tuples
            deterministic: Whether to select deterministically
        
        Returns:
            List of routing actions
        """
        actions = []
        for flow_id, source, destination in flows:
            action = self.infer_routing_action(
                graph, link_states, flow_id, source, destination,
                deterministic=deterministic
            )
            actions.append(action)
        
        return actions
    
    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        self.model.load_checkpoint(checkpoint_path)
        self.model.eval()
    
    def save_model(self, checkpoint_path: str):
        """Save model checkpoint.
        
        Args:
            checkpoint_path: Path to save checkpoint
        """
        self.model.save_checkpoint(checkpoint_path)
    
    def get_model(self) -> GNNAgent:
        """Get underlying model.
        
        Returns:
            GNN agent model
        """
        return self.model


def create_inference_engine(model_path: Optional[str] = None,
                           device: str = 'cpu') -> GNNInferenceEngine:
    """Factory function to create inference engine.
    
    Args:
        model_path: Path to pre-trained model (optional)
        device: Device to run on
    
    Returns:
        Configured inference engine
    """
    engine = GNNInferenceEngine(device=device)
    
    if model_path:
        engine.load_model(model_path)
    
    return engine
