"""
Network simulation engine with packet forwarding and QoS metrics.
Core simulation logic for DRL training and inference.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import copy


@dataclass
class LinkState:
    """State of a network link."""
    source: int
    destination: int
    capacity: float  # Mbps
    current_load: float = 0.0  # Mbps
    latency: float = 5.0  # ms
    packet_loss_rate: float = 0.0
    queue_length: int = 0
    max_queue: int = 1000
    
    @property
    def available_capacity(self) -> float:
        """Get available capacity on link."""
        return self.capacity - self.current_load
    
    @property
    def utilization(self) -> float:
        """Get link utilization percentage."""
        return (self.current_load / self.capacity * 100) if self.capacity > 0 else 0
    
    @property
    def is_congested(self) -> bool:
        """Check if link is congested (>80% utilization)."""
        return self.utilization > 80


@dataclass
class FlowMetrics:
    """Metrics for a network flow."""
    flow_id: int
    source: int
    destination: int
    packets_sent: int = 0
    packets_delivered: int = 0
    packets_dropped: int = 0
    total_latency: float = 0.0
    total_delay: float = 0.0
    start_time: int = 0
    end_time: int = 0
    
    @property
    def packet_loss_rate(self) -> float:
        """Get packet loss rate."""
        if self.packets_sent == 0:
            return 0.0
        return self.packets_dropped / self.packets_sent
    
    @property
    def average_latency(self) -> float:
        """Get average latency."""
        if self.packets_delivered == 0:
            return 0.0
        return self.total_latency / self.packets_delivered
    
    @property
    def average_delay(self) -> float:
        """Get average delay."""
        if self.packets_delivered == 0:
            return 0.0
        return self.total_delay / self.packets_delivered
    
    @property
    def throughput_mbps(self) -> float:
        """Get throughput in Mbps."""
        if self.end_time <= self.start_time:
            return 0.0
        time_duration = self.end_time - self.start_time
        # Assuming 1500 byte packets
        bits_delivered = self.packets_delivered * 1500 * 8
        return (bits_delivered / time_duration) if time_duration > 0 else 0.0


class NetworkSimulator:
    """Main network simulation engine."""
    
    def __init__(self, topology: nx.Graph, link_capacity: float = 100.0):
        """Initialize network simulator.
        
        Args:
            topology: NetworkX graph representing network topology
            link_capacity: Default link capacity in Mbps
        """
        self.topology = topology
        self.num_nodes = topology.number_of_nodes()
        
        # Initialize link states
        self.links: Dict[Tuple[int, int], LinkState] = {}
        self._initialize_links(link_capacity)
        
        # Flow tracking
        self.flows: Dict[int, FlowMetrics] = {}
        self.flow_routes: Dict[int, List[int]] = {}  # flow_id -> path
        
        # Statistics
        self.current_time = 0
        self.total_packets_sent = 0
        self.total_packets_delivered = 0
        self.total_packets_dropped = 0
        
        # Action history
        self.action_history: List[Dict] = []
    
    def _initialize_links(self, link_capacity: float):
        """Initialize all links in topology.
        
        Args:
            link_capacity: Default capacity in Mbps
        """
        for src, dst in self.topology.edges():
            # Bidirectional links
            self.links[(src, dst)] = LinkState(
                source=src,
                destination=dst,
                capacity=link_capacity
            )
            self.links[(dst, src)] = LinkState(
                source=dst,
                destination=src,
                capacity=link_capacity
            )
    
    def route_flow(self, flow_id: int, source: int, destination: int,
                  routing_action: int) -> List[int]:
        """Route a flow using selected action.
        
        Args:
            flow_id: Flow identifier
            source: Source node
            destination: Destination node
            routing_action: Action index (0=shortest hop, 1=min delay, 2=max capacity)
        
        Returns:
            Path from source to destination
        """
        try:
            if routing_action == 0:
                # Shortest hop count path
                path = nx.shortest_path(self.topology, source, destination)
            elif routing_action == 1:
                # Minimum latency path
                path = self._min_latency_path(source, destination)
            elif routing_action == 2:
                # Maximum available capacity path
                path = self._max_capacity_path(source, destination)
            else:
                # Default to shortest path
                path = nx.shortest_path(self.topology, source, destination)
            
            self.flow_routes[flow_id] = path
            return path
        
        except nx.NetworkXNoPath:
            return [source, destination]  # Return direct if no path
        except Exception as e:
            print(f"Error routing flow {flow_id}: {e}")
            return [source, destination]
    
    def _min_latency_path(self, source: int, destination: int) -> List[int]:
        """Find path with minimum latency.
        
        Args:
            source: Source node
            destination: Destination node
        
        Returns:
            Path with minimum total latency
        """
        def latency_weight(u, v):
            return self.links.get((u, v), LinkState(u, v, 100)).latency
        
        try:
            return nx.shortest_path(self.topology, source, destination,
                                   weight=latency_weight)
        except:
            return nx.shortest_path(self.topology, source, destination)
    
    def _max_capacity_path(self, source: int, destination: int) -> List[int]:
        """Find path with maximum available capacity.
        
        Args:
            source: Source node
            destination: Destination node
        
        Returns:
            Path with maximum bottleneck capacity
        """
        def capacity_weight(u, v):
            # Negative for maximum path
            return -self.links.get((u, v), LinkState(u, v, 100)).available_capacity
        
        try:
            return nx.shortest_path(self.topology, source, destination,
                                   weight=capacity_weight)
        except:
            return nx.shortest_path(self.topology, source, destination)
    
    def forward_flow(self, flow_id: int, data_rate: float) -> Tuple[bool, float]:
        """Forward packets for a flow along its routed path.
        
        Args:
            flow_id: Flow identifier
            data_rate: Data rate in Mbps
        
        Returns:
            Tuple of (success, actual_delivered_rate)
        """
        if flow_id not in self.flow_routes:
            return False, 0.0
        
        path = self.flow_routes[flow_id]
        if len(path) < 2:
            return False, 0.0
        
        # Forward along path, constrained by bottleneck link
        delivered_rate = data_rate
        
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            link = self.links.get((u, v))
            
            if link is None:
                return False, 0.0
            
            # Check available capacity
            if link.available_capacity < delivered_rate:
                # Rate-limited by link capacity
                delivered_rate = max(0, link.available_capacity)
            
            # Update link load
            link.current_load += delivered_rate
            
            # Apply packet loss based on congestion
            if link.utilization > 90:
                link.packet_loss_rate = min(0.1, (link.utilization - 90) / 10)
            else:
                link.packet_loss_rate = 0.0
            
            # Update queue
            link.queue_length += int(delivered_rate / 10)
            if link.queue_length > link.max_queue:
                link.queue_length = link.max_queue
                delivered_rate = max(0, delivered_rate - 10)
        
        # Track metrics
        if flow_id in self.flows:
            flow_metrics = self.flows[flow_id]
            packets = int(delivered_rate / 0.01)  # Convert Mbps to packets
            flow_metrics.packets_sent += packets
            flow_metrics.packets_delivered += int(packets * (1 - np.mean(
                [self.links.get((path[i], path[i+1]), LinkState(0, 1, 100)).packet_loss_rate 
                 for i in range(len(path) - 1)]
            )))
            
            # Calculate path latency
            path_latency = sum([self.links.get((path[i], path[i+1]), 
                                              LinkState(0, 1, 100)).latency 
                               for i in range(len(path) - 1)])
            flow_metrics.total_latency += path_latency
            flow_metrics.total_delay += path_latency
        
        self.total_packets_sent += int(delivered_rate / 0.01)
        
        return delivered_rate > 0, delivered_rate
    
    def compute_qos_metrics(self, flow_id: int) -> Dict[str, float]:
        """Compute QoS metrics for a flow.
        
        Args:
            flow_id: Flow identifier
        
        Returns:
            Dictionary of QoS metrics
        """
        if flow_id not in self.flows:
            return {}
        
        metrics = self.flows[flow_id]
        
        return {
            'latency_ms': metrics.average_latency,
            'jitter_ms': 0.0,  # Could compute from latency variance
            'packet_loss': metrics.packet_loss_rate,
            'throughput_mbps': metrics.throughput_mbps,
            'delay_ms': metrics.average_delay
        }
    
    def step(self, flows_to_forward: Dict[int, float], routing_actions: Dict[int, int]):
        """Execute one simulation step.
        
        Args:
            flows_to_forward: Dict of {flow_id: data_rate}
            routing_actions: Dict of {flow_id: action}
        """
        # Reset link loads for this step
        for link in self.links.values():
            link.current_load = 0.0
            link.queue_length = max(0, link.queue_length - 10)
        
        # Route and forward flows
        for flow_id, data_rate in flows_to_forward.items():
            if flow_id in routing_actions:
                action = routing_actions[flow_id]
                
                # Check if we need to re-route
                if flow_id not in self.flow_routes:
                    # Get flow source/destination (would come from flow metadata)
                    # For now, assume it's stored somewhere
                    pass
                
                success, delivered = self.forward_flow(flow_id, data_rate)
        
        # Record action
        self.action_history.append({
            'time': self.current_time,
            'routing_actions': routing_actions.copy(),
            'flows': flows_to_forward.copy()
        })
        
        self.current_time += 1
    
    def reset(self):
        """Reset simulator state."""
        for link in self.links.values():
            link.current_load = 0.0
            link.queue_length = 0
            link.packet_loss_rate = 0.0
        
        self.flows = {}
        self.flow_routes = {}
        self.current_time = 0
        self.total_packets_sent = 0
        self.total_packets_delivered = 0
        self.total_packets_dropped = 0
        self.action_history = []
    
    def get_network_state(self) -> Dict:
        """Get complete network state snapshot.
        
        Returns:
            Dictionary with network state
        """
        link_loads = {str(link): state.current_load 
                     for link, state in self.links.items()}
        link_utils = {str(link): state.utilization 
                     for link, state in self.links.items()}
        
        return {
            'current_time': self.current_time,
            'link_loads': link_loads,
            'link_utilizations': link_utils,
            'num_active_flows': len(self.flows),
            'total_packets_sent': self.total_packets_sent,
            'total_packets_delivered': self.total_packets_delivered
        }
    
    def get_link_state_vector(self) -> np.ndarray:
        """Get link state as feature vector for GNN.
        
        Returns:
            Feature vector of link states
        """
        features = []
        for u, v in self.topology.edges():
            link = self.links.get((u, v))
            if link:
                features.extend([
                    link.utilization / 100.0,  # Normalize to [0, 1]
                    link.queue_length / link.max_queue,
                    link.packet_loss_rate
                ])
        
        return np.array(features)
