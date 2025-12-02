"""
Metrics tracking and collection module.
Collects and analyzes performance metrics during simulation and training.
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime


@dataclass
class QoSMetrics:
    """Quality of Service metrics."""
    latency_ms: float = 0.0  # One-way latency
    jitter_ms: float = 0.0  # Latency variation
    packet_loss_rate: float = 0.0  # PLR (0-1)
    throughput_mbps: float = 0.0  # Goodput
    sla_violation_rate: float = 0.0  # SLA violations
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'latency_ms': self.latency_ms,
            'jitter_ms': self.jitter_ms,
            'packet_loss_rate': self.packet_loss_rate,
            'throughput_mbps': self.throughput_mbps,
            'sla_violation_rate': self.sla_violation_rate
        }


@dataclass
class UtilizationMetrics:
    """Network utilization metrics."""
    avg_link_utilization: float = 0.0  # Percentage
    max_link_utilization: float = 0.0  # Percentage
    j_fairness: float = 0.0  # Jain fairness index
    load_balance_index: float = 0.0  # How well balanced
    congested_links: int = 0  # Number of congested links
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'avg_link_utilization': self.avg_link_utilization,
            'max_link_utilization': self.max_link_utilization,
            'j_fairness': self.j_fairness,
            'load_balance_index': self.load_balance_index,
            'congested_links': self.congested_links
        }


@dataclass
class LearningMetrics:
    """Learning/training metrics."""
    episode_reward: float = 0.0
    cumulative_reward: float = 0.0
    average_action_advantage: float = 0.0
    policy_loss: float = 0.0
    value_loss: float = 0.0
    epsilon: float = 1.0  # Exploration rate
    learning_rate: float = 0.001
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'episode_reward': self.episode_reward,
            'cumulative_reward': self.cumulative_reward,
            'average_action_advantage': self.average_action_advantage,
            'policy_loss': self.policy_loss,
            'value_loss': self.value_loss,
            'epsilon': self.epsilon,
            'learning_rate': self.learning_rate
        }


@dataclass
class OverheadMetrics:
    """Operational overhead metrics."""
    routing_updates: int = 0  # Number of routing changes
    control_messages: int = 0  # Number of control messages
    computation_time_ms: float = 0.0  # Time for decision making
    memory_usage_mb: float = 0.0  # Memory footprint
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'routing_updates': self.routing_updates,
            'control_messages': self.control_messages,
            'computation_time_ms': self.computation_time_ms,
            'memory_usage_mb': self.memory_usage_mb
        }


class MetricsTracker:
    """Tracks metrics during simulation and training."""
    
    def __init__(self, window_size: int = 100):
        """Initialize metrics tracker.
        
        Args:
            window_size: Window size for rolling averages
        """
        self.window_size = window_size
        
        # QoS metrics
        self.qos_metrics: Dict[int, QoSMetrics] = defaultdict(QoSMetrics)
        self.qos_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window_size))
        
        # Utilization metrics
        self.utilization: UtilizationMetrics = UtilizationMetrics()
        self.utilization_history: deque = deque(maxlen=window_size)
        
        # Learning metrics
        self.learning: LearningMetrics = LearningMetrics()
        self.learning_history: deque = deque(maxlen=window_size)
        
        # Overhead metrics
        self.overhead: OverheadMetrics = OverheadMetrics()
        self.overhead_history: deque = deque(maxlen=window_size)
        
        # Episode tracking
        self.current_episode = 0
        self.episode_start_time = datetime.now()
    
    def record_qos_metrics(self, flow_id: int, metrics: QoSMetrics):
        """Record QoS metrics for a flow.
        
        Args:
            flow_id: Flow identifier
            metrics: QoS metrics
        """
        self.qos_metrics[flow_id] = metrics
        self.qos_history[flow_id].append(metrics.to_dict())
    
    def record_utilization(self, metrics: UtilizationMetrics):
        """Record utilization metrics.
        
        Args:
            metrics: Utilization metrics
        """
        self.utilization = metrics
        self.utilization_history.append(metrics.to_dict())
    
    def record_learning_metrics(self, metrics: LearningMetrics):
        """Record learning metrics.
        
        Args:
            metrics: Learning metrics
        """
        self.learning = metrics
        self.learning_history.append(metrics.to_dict())
    
    def record_overhead(self, metrics: OverheadMetrics):
        """Record overhead metrics.
        
        Args:
            metrics: Overhead metrics
        """
        self.overhead = metrics
        self.overhead_history.append(metrics.to_dict())
    
    def get_qos_summary(self) -> Dict[str, float]:
        """Get summary of QoS metrics.
        
        Returns:
            Dictionary of aggregated QoS metrics
        """
        if not self.qos_metrics:
            return {}
        
        metrics_list = list(self.qos_metrics.values())
        
        return {
            'avg_latency_ms': np.mean([m.latency_ms for m in metrics_list]),
            'max_latency_ms': np.max([m.latency_ms for m in metrics_list]),
            'avg_jitter_ms': np.mean([m.jitter_ms for m in metrics_list]),
            'avg_packet_loss': np.mean([m.packet_loss_rate for m in metrics_list]),
            'avg_throughput_mbps': np.mean([m.throughput_mbps for m in metrics_list]),
            'avg_sla_violation': np.mean([m.sla_violation_rate for m in metrics_list]),
            'num_flows': len(metrics_list)
        }
    
    def get_utilization_summary(self) -> Dict[str, float]:
        """Get summary of utilization metrics.
        
        Returns:
            Dictionary of utilization metrics
        """
        return self.utilization.to_dict()
    
    def get_learning_summary(self) -> Dict[str, float]:
        """Get summary of learning metrics.
        
        Returns:
            Dictionary of learning metrics
        """
        return self.learning.to_dict()
    
    def get_overhead_summary(self) -> Dict[str, float]:
        """Get summary of overhead metrics.
        
        Returns:
            Dictionary of overhead metrics
        """
        return self.overhead.to_dict()
    
    def get_episode_summary(self) -> Dict[str, Any]:
        """Get complete episode summary.
        
        Returns:
            Dictionary with all metrics summaries
        """
        return {
            'episode': self.current_episode,
            'timestamp': datetime.now().isoformat(),
            'qos': self.get_qos_summary(),
            'utilization': self.get_utilization_summary(),
            'learning': self.get_learning_summary(),
            'overhead': self.get_overhead_summary()
        }
    
    def get_rolling_average(self, metric_name: str, metric_type: str = 'learning') -> float:
        """Get rolling average of metric.
        
        Args:
            metric_name: Name of metric
            metric_type: Type of metric (learning, qos, utilization, overhead)
        
        Returns:
            Rolling average value
        """
        if metric_type == 'learning' and self.learning_history:
            values = [m.get(metric_name, 0) for m in self.learning_history]
            return np.mean(values) if values else 0.0
        elif metric_type == 'utilization' and self.utilization_history:
            values = [m.get(metric_name, 0) for m in self.utilization_history]
            return np.mean(values) if values else 0.0
        elif metric_type == 'overhead' and self.overhead_history:
            values = [m.get(metric_name, 0) for m in self.overhead_history]
            return np.mean(values) if values else 0.0
        
        return 0.0
    
    def reset_episode(self):
        """Reset for new episode."""
        self.current_episode += 1
        self.qos_metrics.clear()
        self.episode_start_time = datetime.now()
    
    def export_metrics(self, filepath: str):
        """Export all metrics to file.
        
        Args:
            filepath: Path to export to
        """
        data = {
            'episode': self.current_episode,
            'timestamp': datetime.now().isoformat(),
            'qos_history': [dict(m) for m in self.qos_history.values() if m],
            'utilization_history': list(self.utilization_history),
            'learning_history': list(self.learning_history),
            'overhead_history': list(self.overhead_history)
        }
        
        import json
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def compute_aggregate_metrics(self) -> Dict[str, float]:
        """Compute aggregate metrics across all flows.
        
        Returns:
            Dictionary of aggregated metrics
        """
        all_metrics = {}
        all_metrics.update(self.get_qos_summary())
        all_metrics.update(self.get_utilization_summary())
        all_metrics.update(self.get_learning_summary())
        all_metrics.update(self.get_overhead_summary())
        
        return all_metrics


class MetricsAggregator:
    """Aggregates metrics across multiple episodes/runs."""
    
    def __init__(self):
        """Initialize metrics aggregator."""
        self.episode_metrics: List[Dict] = []
    
    def add_episode(self, metrics: Dict):
        """Add episode metrics.
        
        Args:
            metrics: Episode metrics dictionary
        """
        self.episode_metrics.append(metrics)
    
    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics across episodes.
        
        Returns:
            Dictionary of statistics by metric
        """
        if not self.episode_metrics:
            return {}
        
        stats = {}
        
        # Collect all metric keys
        all_keys = set()
        for metrics in self.episode_metrics:
            all_keys.update(metrics.keys())
        
        # Compute statistics for each metric
        for key in all_keys:
            values = []
            for metrics in self.episode_metrics:
                if key in metrics and isinstance(metrics[key], (int, float)):
                    values.append(metrics[key])
            
            if values:
                stats[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values)
                }
        
        return stats
    
    def reset(self):
        """Reset aggregator."""
        self.episode_metrics = []
