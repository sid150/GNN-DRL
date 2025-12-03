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
        
        Simplified to 2D: [load, queue_length] to match notebook architecture.
        
        Args:
            graph: Network topology graph
            link_states: Current link state information
        
        Returns:
            Node feature tensor (num_nodes, 2)
        """
        features = []
        nodes = sorted(list(graph.nodes()))
        
        for node in nodes:
            # Calculate total load and queue for this node
            total_load = 0.0
            total_queue = 0
            
            for (u, v), state in link_states.items():
                if u == node:
                    total_load += state.current_load
                    total_queue += state.queue_length
            
            # Normalize: load by 1000 Mbps, queue by 100
            node_features = [
                total_load / 1000.0,
                total_queue / 100.0
            ]
            
            features.append(node_features)
        
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
    
    def encode_flow_state(self, flow_id: int, source: int, destination: int, 
                         link_states: Dict = None) -> torch.Tensor:
        """Encode flow state for agent input.
        
        Encodes 6 features: [src_idx, dst_idx, volume, priority, avg_load, congestion]
        
        Args:
            flow_id: Flow identifier
            source: Source node
            destination: Destination node
            link_states: Current link states for computing avg_load and congestion
        
        Returns:
            Encoded flow state tensor (6,)
        """
        # Normalize source and destination indices
        src_idx_norm = source / max(self.num_nodes, 1)
        dst_idx_norm = destination / max(self.num_nodes, 1)
        
        # Default flow volume and priority (can be enhanced with actual flow data)
        volume_norm = 0.5  # Assume medium volume
        priority_norm = 0.5  # Assume medium priority
        
        # Compute average load and congestion if link_states provided
        avg_load = 0.0
        congestion = 0.0
        if link_states:
            loads = [state.current_load for state in link_states.values()]
            if loads:
                avg_load = np.mean(loads) / 1000.0
                congestion = len([l for l in loads if l > 500]) / len(loads)
        
        flow_state = [
            src_idx_norm,
            dst_idx_norm,
            volume_norm,
            priority_norm,
            avg_load,
            congestion
        ]
        return torch.FloatTensor(flow_state)


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
        flow_state = self.preprocessor.encode_flow_state(flow_id, source, destination, link_states)
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
                deterministic=deterministic,
                epsilon=0.0  # No exploration during inference
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
