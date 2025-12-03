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
        
        # Simplified GCN layers (2 layers like notebook)
        self.gcn1 = GCNLayer(config.node_feature_dim, config.hidden_dim, dropout=0.0)
        self.gcn2 = GCNLayer(config.hidden_dim, config.hidden_dim, dropout=0.0)
        
        # Flow encoder (6 features: src_idx, dst_idx, volume, priority, avg_load, congestion)
        self.flow_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, config.hidden_dim)
        )
        
        # Policy network head (simplified)
        self.policy_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_actions)
        )
        
        # Value network head (simplified)
        self.value_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), 
                                         lr=config.learning_rate)
        
        # Experience tracking
        self.experience_buffer = {
            'node_features': [],
            'flow_states': [],
            'edge_indices': [],
            'actions': [],
            'rewards': [],
            'next_node_features': [],
            'next_flow_states': [],
            'next_edge_indices': [],
            'dones': []
        }
        self.episode_rewards = []
        self.episode_actions = []
    
    def forward(self, node_features: torch.Tensor, edge_index: torch.Tensor,
               flow_state: torch.Tensor, adj_matrix: torch.Tensor = None) \
               -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through agent.
        
        Args:
            node_features: Node feature matrix (num_nodes, node_feature_dim)
            edge_index: Edge indices (2, num_edges)
            flow_state: Current flow state (6 features)
            adj_matrix: Adjacency matrix (num_nodes, num_nodes) - optional
        
        Returns:
            Tuple of (action_logits, value_estimate)
        """
        # GCN forward pass (2 layers)
        if adj_matrix is None:
            # Build adjacency from edge_index
            num_nodes = node_features.size(0)
            adj_matrix = torch.zeros((num_nodes, num_nodes), device=node_features.device)
            if edge_index.size(1) > 0:
                adj_matrix[edge_index[0], edge_index[1]] = 1.0
        
        x = F.relu(self.gcn1(node_features, adj_matrix))
        x = F.relu(self.gcn2(x, adj_matrix))
        
        # Aggregate node embeddings
        graph_embedding = torch.mean(x, dim=0)
        
        # Encode flow state (6 features)
        flow_embedding = self.flow_encoder(flow_state.float())
        
        # Combine embeddings (add instead of concatenate like notebook)
        combined = graph_embedding + flow_embedding
        
        # Get policy and value
        action_logits = self.policy_head(combined)
        value = self.value_head(combined)
        
        return action_logits, value
    
    def select_action(self, node_features: torch.Tensor, edge_index: torch.Tensor,
                     flow_state: torch.Tensor, adj_matrix: torch.Tensor = None,
                     deterministic: bool = False, epsilon: float = 0.1) -> int:
        """Select action for current state.
        
        Args:
            node_features: Node feature matrix
            edge_index: Edge indices
            flow_state: Flow state
            adj_matrix: Adjacency matrix
            deterministic: Whether to select action deterministically
            epsilon: Exploration rate (epsilon-greedy)
        
        Returns:
            Selected action index
        """
        with torch.no_grad():
            # Ensure flow_state is tensor
            if isinstance(flow_state, np.ndarray):
                flow_state = torch.tensor(flow_state, dtype=torch.float32)
            
            logits, _ = self.forward(node_features, edge_index, flow_state, adj_matrix)
            probabilities = F.softmax(logits, dim=0)
            
            if deterministic:
                action = probabilities.argmax().item()
            elif np.random.random() < epsilon:
                # Explore
                action = np.random.randint(self.config.num_actions)
            else:
                # Exploit
                action = probabilities.argmax().item()
            
            self.episode_actions.append(action)
            return action
    
    def store_experience(self, node_features, flow_state, edge_index, action: int,
                        reward: float, next_node_features, next_flow_state,
                        next_edge_index, done: bool):
        """Store experience in buffer.
        
        Args:
            node_features: Current node features
            flow_state: Current flow state
            edge_index: Current edge index
            action: Action taken
            reward: Reward received
            next_node_features: Next node features
            next_flow_state: Next flow state
            next_edge_index: Next edge index
            done: Whether episode ended
        """
        self.experience_buffer['node_features'].append(node_features)
        self.experience_buffer['flow_states'].append(flow_state)
        self.experience_buffer['edge_indices'].append(edge_index)
        self.experience_buffer['actions'].append(action)
        self.experience_buffer['rewards'].append(reward)
        self.experience_buffer['next_node_features'].append(next_node_features)
        self.experience_buffer['next_flow_states'].append(next_flow_state)
        self.experience_buffer['next_edge_indices'].append(next_edge_index)
        self.experience_buffer['dones'].append(done)
    
    def record_episode(self, episode_reward: float):
        """Record episode completion.
        
        Args:
            episode_reward: Total episode reward
        """
        self.episode_rewards.append(episode_reward)
        self.episode_actions = []
    
    def update_policy(self, batch_size: int = 32, gamma: float = 0.99) -> Dict[str, float]:
        """Update policy using experience buffer (notebook approach).
        
        Args:
            batch_size: Batch size for training
            gamma: Discount factor
        
        Returns:
            Dictionary of loss values
        """
        if len(self.experience_buffer['actions']) < batch_size:
            return {'policy_loss': 0.0, 'value_loss': 0.0, 'total_loss': 0.0}
        
        indices = np.random.choice(len(self.experience_buffer['actions']),
                                   batch_size, replace=False)
        
        policy_losses = []
        value_losses = []
        
        device = next(self.parameters()).device
        
        for idx in indices:
            node_features = self.experience_buffer['node_features'][idx].to(device)
            flow_state = self.experience_buffer['flow_states'][idx]
            if isinstance(flow_state, np.ndarray):
                flow_state = torch.tensor(flow_state, dtype=torch.float32)
            edge_index = self.experience_buffer['edge_indices'][idx].to(device)
            action = self.experience_buffer['actions'][idx]
            reward = self.experience_buffer['rewards'][idx]
            next_node_features = self.experience_buffer['next_node_features'][idx].to(device)
            next_flow_state = self.experience_buffer['next_flow_states'][idx]
            if isinstance(next_flow_state, np.ndarray):
                next_flow_state = torch.tensor(next_flow_state, dtype=torch.float32)
            next_edge_index = self.experience_buffer['next_edge_indices'][idx].to(device)
            done = self.experience_buffer['dones'][idx]
            
            # Compute target value
            with torch.no_grad():
                _, next_value = self.forward(next_node_features, next_edge_index, next_flow_state)
                next_value = next_value.item() if not done else 0.0
                td_target = reward + gamma * next_value
            
            # Forward pass
            policy_logits, value = self.forward(node_features, edge_index, flow_state)
            value = value.squeeze()
            
            # Compute losses
            value_loss = F.mse_loss(value, torch.tensor(td_target, dtype=torch.float32, device=device))
            policy_loss = -F.log_softmax(policy_logits, dim=0)[action]
            
            policy_losses.append(policy_loss)
            value_losses.append(value_loss)
        
        # Combined loss
        total_policy_loss = torch.stack(policy_losses).mean()
        total_value_loss = torch.stack(value_losses).mean()
        total_loss = total_policy_loss + total_value_loss
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return {
            'policy_loss': total_policy_loss.item(),
            'value_loss': total_value_loss.item(),
            'total_loss': total_loss.item()
        }
    
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
