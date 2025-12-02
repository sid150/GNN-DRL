"""
Graph Neural Network agent for routing decisions.
Implements GCN-based policy and value networks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class GNNConfig:
    """Configuration for GNN agent."""
    node_feature_dim: int = 128
    edge_feature_dim: int = 64
    hidden_dim: int = 256
    num_gnn_layers: int = 2
    num_actions: int = 3  # Routing actions
    dropout_rate: float = 0.1
    learning_rate: float = 0.001


class GCNLayer(nn.Module):
    """Graph Convolutional Network layer."""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.0):
        """Initialize GCN layer.
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.dropout = nn.Dropout(dropout)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Reset parameters."""
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """Forward pass through GCN layer.
        
        Args:
            x: Node features (num_nodes, in_features)
            adj: Adjacency matrix (num_nodes, num_nodes)
        
        Returns:
            Node embeddings (num_nodes, out_features)
        """
        x = self.dropout(x)
        x = torch.matmul(x, self.weight)
        
        # Normalize adjacency matrix
        if adj.sum() > 0:
            adj_norm = self._normalize_adjacency(adj)
            x = torch.matmul(adj_norm, x)
        
        x = x + self.bias
        return F.relu(x)
    
    @staticmethod
    def _normalize_adjacency(adj: torch.Tensor) -> torch.Tensor:
        """Normalize adjacency matrix using symmetric normalization.
        
        Args:
            adj: Adjacency matrix
        
        Returns:
            Normalized adjacency matrix
        """
        # Add self-loops
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Compute degree matrix
        degree = adj.sum(dim=1)
        degree = torch.pow(degree, -0.5)
        degree[torch.isinf(degree)] = 0.0
        
        # Symmetric normalization: D^-0.5 * A * D^-0.5
        degree_matrix = torch.diag(degree)
        adj_norm = torch.matmul(torch.matmul(degree_matrix, adj), degree_matrix)
        
        return adj_norm


class FlowEncoder(nn.Module):
    """Encodes flow state information."""
    
    def __init__(self, hidden_dim: int):
        """Initialize flow encoder.
        
        Args:
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),  # flow_id, src, dst
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
    
    def forward(self, flow_state: torch.Tensor) -> torch.Tensor:
        """Encode flow state.
        
        Args:
            flow_state: Flow state tensor
        
        Returns:
            Encoded flow representation
        """
        return self.mlp(flow_state)


class GNNAgent(nn.Module):
    """GCN-based agent for routing decisions."""
    
    def __init__(self, config: GNNConfig):
        """Initialize GNN agent.
        
        Args:
            config: GNN configuration
        """
        super().__init__()
        self.config = config
        
        # GCN layers for node embedding
        self.gcn_layers = nn.ModuleList([
            GCNLayer(config.node_feature_dim if i == 0 else config.hidden_dim,
                    config.hidden_dim,
                    dropout=config.dropout_rate)
            for i in range(config.num_gnn_layers)
        ])
        
        # Flow encoder
        self.flow_encoder = FlowEncoder(config.hidden_dim)
        
        # Policy network head
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.hidden_dim // 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.num_actions)
        )
        
        # Value network head
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim + config.hidden_dim // 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=config.learning_rate)
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
               flow_state: torch.Tensor, adj_matrix: torch.Tensor) \
               -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through agent.
        
        Args:
            node_features: Node feature matrix (num_nodes, node_feature_dim)
            edge_index: Edge indices (2, num_edges)
            flow_state: Current flow state (batch, 3) [flow_id, src, dst]
            adj_matrix: Adjacency matrix (num_nodes, num_nodes)
        
        Returns:
            Tuple of (action_logits, value_estimate)
        """
        # GCN forward pass
        x = node_features
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, adj_matrix)
        
        # Aggregate node embeddings
        graph_embedding = torch.mean(x, dim=0)
        
        # Encode flow state
        flow_embedding = self.flow_encoder(flow_state.float())
        
        # Combine graph and flow embeddings
        combined = torch.cat([graph_embedding.unsqueeze(0), 
                            flow_embedding], dim=1)
        
        # Get policy and value
        action_logits = self.policy_head(combined)
        value = self.value_head(combined)
        
        return action_logits, value
    
    def select_action(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                     flow_state: torch.Tensor, adj_matrix: torch.Tensor,
                     deterministic: bool = False) -> int:
        """Select action for current state.
        
        Args:
            node_features: Node feature matrix
            edge_index: Edge indices
            flow_state: Flow state
            adj_matrix: Adjacency matrix
            deterministic: Whether to select action deterministically
        
        Returns:
            Selected action index
        """
        with torch.no_grad():
            logits, _ = self.forward(node_features, edge_index, flow_state, adj_matrix)
            
            if deterministic:
                action = torch.argmax(logits, dim=1).item()
            else:
                probs = F.softmax(logits, dim=1)
                action = torch.multinomial(probs, 1).item()
            
            return action
    
    def update_policy(self, states: List[Dict], actions: np.ndarray,
                     rewards: np.ndarray, next_states: List[Dict],
                     dones: np.ndarray, gamma: float = 0.99) -> Dict[str, float]:
        """Update policy using experience.
        
        Args:
            states: List of state dicts
            actions: Actions taken (array)
            rewards: Rewards received (array)
            next_states: Next states (array)
            dones: Whether episode ended (array)
            gamma: Discount factor
        
        Returns:
            Dictionary of loss values
        """
        losses = {}
        
        # Convert to tensors
        actions_tensor = torch.LongTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)
        dones_tensor = torch.FloatTensor(dones)
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for i, state in enumerate(states):
            if i >= len(next_states):
                continue
            
            # Get current state value and policy
            logits, value = self.forward(
                state['node_features'],
                state['edge_index'],
                state['flow_state'],
                state['adjacency_matrix']
            )
            
            # Get next state value (for bootstrapping)
            with torch.no_grad():
                next_logits, next_value = self.forward(
                    next_states[i]['node_features'],
                    next_states[i]['edge_index'],
                    next_states[i]['flow_state'],
                    next_states[i]['adjacency_matrix']
                )
            
            # Compute advantage
            target_value = rewards_tensor[i] + gamma * next_value.squeeze() * (1 - dones_tensor[i])
            advantage = target_value - value.squeeze()
            
            # Policy loss (cross-entropy)
            policy_loss = F.cross_entropy(logits, actions_tensor[i:i+1])
            
            # Value loss (MSE)
            value_loss = F.mse_loss(value.squeeze(), target_value)
            
            # Entropy regularization
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum()
            
            # Combined loss
            loss = policy_loss + value_loss - 0.01 * entropy
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        losses['policy_loss'] = total_policy_loss / max(len(states), 1)
        losses['value_loss'] = total_value_loss / max(len(states), 1)
        
        return losses
    
    def save_checkpoint(self, filepath: str):
        """Save model checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'model_state': self.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': self.config.__dict__
        }, filepath)
    
    def load_checkpoint(self, filepath: str):
        """Load model checkpoint.
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, weights_only=False, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model_state' in checkpoint:
            # New format
            self.load_state_dict(checkpoint['model_state'])
            if 'optimizer_state' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        elif 'agent_state' in checkpoint:
            # Old format from different implementation - skip loading
            # The model will use random initialization but inference will work
            print("Warning: Old checkpoint format detected. Using model with random initialization.")
            print("For best results, please train a new model with: python main.py train --episodes 100")
        else:
            # Try to load as direct state dict
            try:
                self.load_state_dict(checkpoint)
            except Exception as e:
                print(f"Warning: Could not load checkpoint: {e}")
                print("Using model with random initialization.")


def create_gnn_agent(node_dim: int = 128, hidden_dim: int = 256,
                     num_layers: int = 2, num_actions: int = 3,
                     learning_rate: float = 0.001) -> GNNAgent:
    """Factory function to create GNN agent.
    
    Args:
        node_dim: Node feature dimension
        hidden_dim: Hidden dimension
        num_layers: Number of GCN layers
        num_actions: Number of action classes
        learning_rate: Learning rate
    
    Returns:
        Initialized GNNAgent
    """
    config = GNNConfig(
        node_feature_dim=node_dim,
        hidden_dim=hidden_dim,
        num_gnn_layers=num_layers,
        num_actions=num_actions,
        learning_rate=learning_rate
    )
    return GNNAgent(config)
