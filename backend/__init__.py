"""
GNN-DRL Backend Application

Modular backend for network routing optimization using Graph Neural Networks and Deep Reinforcement Learning.
"""

__version__ = "1.0.0"
__author__ = "GNN-DRL Team"

from .app_orchestrator import NetworkRoutingSimulator
from .topology_manager import NetworkTopologyBuilder
from .traffic_generator import TrafficDemandGenerator
from .inference_engine import GNNInferenceEngine
from .learning_module import OnlineLearningModule
from .version_manager import ModelVersionManager
from .metrics_tracker import MetricsTracker

__all__ = [
    'NetworkRoutingSimulator',
    'NetworkTopologyBuilder',
    'TrafficDemandGenerator',
    'GNNInferenceEngine',
    'OnlineLearningModule',
    'ModelVersionManager',
    'MetricsTracker'
]
