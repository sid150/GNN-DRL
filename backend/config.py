"""
Configuration management for GNN-DRL backend.
Centralized configuration with environment-based overrides.
"""

import os
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class RoutingStrategy(Enum):
    """Available routing strategies."""
    SHORTEST_HOP = 0
    MIN_DELAY = 1
    MAX_CAPACITY = 2


class TrafficPattern(Enum):
    """Available traffic patterns."""
    UNIFORM = "uniform"
    HOTSPOT = "hotspot"
    BIMODAL = "bimodal"


@dataclass
class NetworkConfig:
    """Network simulation configuration."""
    num_nodes: int = 14
    traffic_matrix_size: int = 10
    max_flows_per_node: int = 5
    link_capacity: float = 100.0  # Mbps
    packet_size: int = 1500  # bytes
    simulation_duration: int = 100  # time steps
    
    @property
    def link_bandwidth_mbps(self) -> float:
        """Link bandwidth in Mbps."""
        return self.link_capacity
    
    @property
    def link_delay_base(self) -> float:
        """Base link delay in ms."""
        return 5.0


@dataclass
class GNNConfig:
    """Graph Neural Network configuration."""
    node_feature_dim: int = 128
    edge_feature_dim: int = 64
    hidden_dim: int = 256
    num_gnn_layers: int = 2
    num_classes: int = 3  # Routing actions: shortest, min-delay, max-capacity
    dropout_rate: float = 0.1
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    
    # Model paths
    model_checkpoint_dir: str = "./models/checkpoints"
    best_model_path: str = "./models/best_model.pt"
    latest_model_path: str = "./models/latest_model.pt"


@dataclass
class TrainingConfig:
    """Training configuration."""
    num_episodes: int = 500
    num_topologies: int = 402
    batch_size: int = 32
    experience_buffer_size: int = 10000
    target_update_frequency: int = 10
    gamma: float = 0.99  # Discount factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    max_grad_norm: float = 1.0
    
    @property
    def total_flows_per_episode(self) -> int:
        """Calculate total flows per episode."""
        return 20  # flows per topology


@dataclass
class MetricsConfig:
    """Metrics collection configuration."""
    track_qos: bool = True
    track_utilization: bool = True
    track_learning: bool = True
    track_overhead: bool = True
    
    # Metric thresholds
    max_latency_sla: float = 50.0  # ms
    min_throughput_sla: float = 10.0  # Mbps
    max_jitter_sla: float = 20.0  # ms
    
    # Metric collection intervals
    metric_collection_interval: int = 1  # Every step
    metric_storage_batch_size: int = 100  # Store metrics in batches


@dataclass
class APIConfig:
    """REST API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    workers: int = 4
    
    # API routes
    inference_endpoint: str = "/api/v1/inference"
    topology_endpoint: str = "/api/v1/topology"
    metrics_endpoint: str = "/api/v1/metrics"
    models_endpoint: str = "/api/v1/models"
    
    # WebSocket configuration
    websocket_enabled: bool = True
    websocket_port: int = 8001


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_type: str = "sqlite"  # or "postgresql"
    db_path: str = "./data/metrics.db"
    
    # PostgreSQL settings (if db_type == "postgresql")
    pg_host: str = "localhost"
    pg_port: int = 5432
    pg_database: str = "gnn_drl"
    pg_user: str = "postgres"
    pg_password: str = ""
    
    # Storage settings
    enable_persistence: bool = True
    retention_days: int = 30
    backup_frequency: int = 24  # hours


class Config:
    """Main configuration class."""
    
    def __init__(self, env: str = "development"):
        """Initialize configuration.
        
        Args:
            env: Environment type (development, testing, production)
        """
        self.env = env
        
        # Load environment-specific overrides
        self._load_environment_overrides()
        
        # Sub-configurations
        self.network = NetworkConfig()
        self.gnn = GNNConfig()
        self.training = TrainingConfig()
        self.metrics = MetricsConfig()
        self.api = APIConfig()
        self.database = DatabaseConfig()
        
        # Apply environment-specific adjustments
        self._apply_environment_config()
    
    def _load_environment_overrides(self):
        """Load overrides from environment variables."""
        os.makedirs("./models/checkpoints", exist_ok=True)
        os.makedirs("./data", exist_ok=True)
    
    def _apply_environment_config(self):
        """Apply environment-specific configuration."""
        if self.env == "production":
            self.api.debug = False
            self.api.workers = 8
            self.database.enable_persistence = True
            self.metrics.track_qos = True
        elif self.env == "testing":
            self.training.num_episodes = 5
            self.training.num_topologies = 10
            self.api.debug = True
            self.database.db_path = ":memory:"
        else:  # development
            self.api.debug = True
            self.training.num_episodes = 10
    
    def to_dict(self) -> Dict:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            'network': self.network.__dict__,
            'gnn': self.gnn.__dict__,
            'training': self.training.__dict__,
            'metrics': self.metrics.__dict__,
            'api': self.api.__dict__,
            'database': self.database.__dict__,
        }
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"GNN-DRL Config ({self.env})"


# Global configuration instance
_config = None


def get_config(env: str = None) -> Config:
    """Get global configuration instance.
    
    Args:
        env: Environment type (optional, uses current if None)
    
    Returns:
        Config instance
    """
    global _config
    if _config is None:
        env = env or os.getenv("GNN_DRL_ENV", "development")
        _config = Config(env)
    return _config


def reset_config():
    """Reset global configuration instance (useful for testing)."""
    global _config
    _config = None
