"""
Traffic generation and demand management module.
Handles creation of realistic traffic patterns for network simulation.
"""

import numpy as np
import networkx as nx
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class TrafficPattern(Enum):
    """Available traffic patterns."""
    UNIFORM = "uniform"
    HOTSPOT = "hotspot"
    BIMODAL = "bimodal"
    GRAVITY = "gravity"


@dataclass
class Flow:
    """Represents a network flow."""
    flow_id: int
    source: int
    destination: int
    arrival_time: int
    duration: int
    data_rate: float  # Mbps
    packet_size: int = 1500  # bytes
    
    @property
    def is_active(self, current_time: int) -> bool:
        """Check if flow is active at given time."""
        return self.arrival_time <= current_time < self.arrival_time + self.duration
    
    def get_active_duration(self, current_time: int) -> int:
        """Get remaining duration of flow."""
        if not self.is_active(current_time):
            return 0
        return self.arrival_time + self.duration - current_time


@dataclass
class TrafficDemand:
    """Represents traffic demand between two nodes."""
    source: int
    destination: int
    demand_mbps: float
    priority: int = 0
    sla_latency_ms: float = 50.0


class TrafficDemandGenerator:
    """Generates network traffic demands and flows."""
    
    def __init__(self, num_nodes: int, seed: Optional[int] = None):
        """Initialize traffic generator.
        
        Args:
            num_nodes: Number of nodes in network
            seed: Random seed for reproducibility
        """
        self.num_nodes = num_nodes
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
        
        self.flows: List[Flow] = []
        self.flow_counter = 0
    
    def generate_static_demands(self, pattern: str = "uniform",
                               total_demand: float = 1000.0,
                               graph: Optional[nx.Graph] = None) -> List[TrafficDemand]:
        """Generate static traffic demands.
        
        Args:
            pattern: Traffic pattern type (uniform, hotspot, bimodal, gravity)
            total_demand: Total demand in Mbps across all flows
            graph: Optional network topology (for gravity model)
        
        Returns:
            List of TrafficDemand objects
        """
        demands = []
        
        if pattern == "uniform":
            demands = self._uniform_pattern(total_demand)
        elif pattern == "hotspot":
            demands = self._hotspot_pattern(total_demand)
        elif pattern == "bimodal":
            demands = self._bimodal_pattern(total_demand)
        elif pattern == "gravity":
            if graph is None:
                raise ValueError("Graph required for gravity model")
            demands = self._gravity_pattern(total_demand, graph)
        
        return demands
    
    def _uniform_pattern(self, total_demand: float) -> List[TrafficDemand]:
        """Generate uniform traffic pattern.
        
        Args:
            total_demand: Total demand in Mbps
        
        Returns:
            List of traffic demands
        """
        num_pairs = self.num_nodes * (self.num_nodes - 1)
        demand_per_pair = total_demand / num_pairs if num_pairs > 0 else 0
        
        demands = []
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src != dst:
                    demands.append(TrafficDemand(
                        source=src,
                        destination=dst,
                        demand_mbps=demand_per_pair,
                        priority=0
                    ))
        
        return demands
    
    def _hotspot_pattern(self, total_demand: float) -> List[TrafficDemand]:
        """Generate hotspot traffic pattern.
        
        Args:
            total_demand: Total demand in Mbps
        
        Returns:
            List of traffic demands
        """
        demands = []
        
        # Select 2-3 hotspots
        num_hotspots = np.random.randint(2, 4)
        hotspots = np.random.choice(self.num_nodes, num_hotspots, replace=False)
        
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src != dst:
                    # Higher demand to/from hotspots
                    if src in hotspots or dst in hotspots:
                        demand = total_demand * 0.7 / (num_hotspots * self.num_nodes)
                    else:
                        demand = total_demand * 0.3 / (self.num_nodes * (self.num_nodes - 1))
                    
                    demands.append(TrafficDemand(
                        source=src,
                        destination=dst,
                        demand_mbps=demand,
                        priority=1 if src in hotspots or dst in hotspots else 0
                    ))
        
        return demands
    
    def _bimodal_pattern(self, total_demand: float) -> List[TrafficDemand]:
        """Generate bimodal traffic pattern (two classes of flows).
        
        Args:
            total_demand: Total demand in Mbps
        
        Returns:
            List of traffic demands
        """
        demands = []
        
        # Split into elephant (heavy) and mice (light) flows
        num_pairs = self.num_nodes * (self.num_nodes - 1)
        
        # 80% traffic on 20% of flows (elephant flows)
        num_elephant_flows = max(1, int(0.2 * num_pairs))
        elephant_demand = total_demand * 0.8 / num_elephant_flows
        mouse_demand = total_demand * 0.2 / (num_pairs - num_elephant_flows)
        
        flow_pairs = [(src, dst) for src in range(self.num_nodes) 
                     for dst in range(self.num_nodes) if src != dst]
        elephant_pairs = set(np.random.choice(len(flow_pairs), num_elephant_flows, replace=False))
        
        for idx, (src, dst) in enumerate(flow_pairs):
            demand = elephant_demand if idx in elephant_pairs else mouse_demand
            demands.append(TrafficDemand(
                source=src,
                destination=dst,
                demand_mbps=demand,
                priority=1 if idx in elephant_pairs else 0
            ))
        
        return demands
    
    def _gravity_pattern(self, total_demand: float, graph: nx.Graph) -> List[TrafficDemand]:
        """Generate gravity model traffic pattern.
        
        Based on network topology - demands proportional to node degrees.
        
        Args:
            total_demand: Total demand in Mbps
            graph: Network topology
        
        Returns:
            List of traffic demands
        """
        demands = []
        
        # Get node degrees (as proxy for node traffic volume)
        degrees = dict(graph.degree())
        total_degree = sum(degrees.values())
        
        for src in range(self.num_nodes):
            for dst in range(self.num_nodes):
                if src != dst:
                    # Demand proportional to product of node degrees
                    demand = (total_demand * 
                             (degrees.get(src, 1) * degrees.get(dst, 1)) / 
                             (total_degree ** 2))
                    demands.append(TrafficDemand(
                        source=src,
                        destination=dst,
                        demand_mbps=demand,
                        priority=0
                    ))
        
        return demands
    
    def generate_dynamic_flows(self, num_flows: int, simulation_duration: int,
                              pattern: str = "uniform",
                              arrival_rate: float = 1.0) -> List[Flow]:
        """Generate dynamic flows with temporal characteristics.
        
        Args:
            num_flows: Number of flows to generate
            simulation_duration: Duration of simulation
            pattern: Flow arrival pattern (uniform, poisson, burst)
            arrival_rate: Average flow arrival rate (flows per time unit)
        
        Returns:
            List of Flow objects
        """
        flows = []
        
        for i in range(num_flows):
            # Random source and destination
            src = np.random.randint(0, self.num_nodes)
            dst = np.random.randint(0, self.num_nodes)
            while dst == src:
                dst = np.random.randint(0, self.num_nodes)
            
            # Arrival time based on pattern
            if pattern == "uniform":
                arrival = int(np.random.uniform(0, simulation_duration))
            elif pattern == "poisson":
                # Poisson process approximation
                inter_arrival = int(np.random.exponential(1.0 / arrival_rate))
                arrival = int(sum(np.random.exponential(1.0 / arrival_rate, i + 1)))
                arrival = min(arrival, simulation_duration - 1)
            elif pattern == "burst":
                # Burst arrival pattern
                arrival = int(np.random.exponential(simulation_duration / 10))
                arrival = min(arrival, simulation_duration - 1)
            else:
                arrival = 0
            
            # Duration and data rate
            duration = int(np.random.exponential(scale=20)) + 1
            duration = min(duration, simulation_duration - arrival)
            data_rate = np.random.uniform(10, 100)  # 10-100 Mbps
            
            flow = Flow(
                flow_id=self.flow_counter,
                source=src,
                destination=dst,
                arrival_time=arrival,
                duration=duration,
                data_rate=data_rate
            )
            
            flows.append(flow)
            self.flow_counter += 1
        
        return flows
    
    def get_active_flows(self, flows: List[Flow], current_time: int) -> List[Flow]:
        """Get all active flows at current time.
        
        Args:
            flows: List of all flows
            current_time: Current simulation time
        
        Returns:
            List of active flows
        """
        return [f for f in flows 
               if f.arrival_time <= current_time < f.arrival_time + f.duration]
    
    def get_flow_demand_matrix(self, flows: List[Flow], current_time: int,
                              num_nodes: int) -> np.ndarray:
        """Get traffic demand matrix for active flows.
        
        Args:
            flows: List of flows
            current_time: Current simulation time
            num_nodes: Number of nodes
        
        Returns:
            Demand matrix (num_nodes x num_nodes)
        """
        demand_matrix = np.zeros((num_nodes, num_nodes))
        
        for flow in self.get_active_flows(flows, current_time):
            demand_matrix[flow.source, flow.destination] += flow.data_rate
        
        return demand_matrix
    
    def reset(self):
        """Reset traffic generator state."""
        self.flows = []
        self.flow_counter = 0


class TrafficMatrix:
    """Manages traffic matrix for network demands."""
    
    def __init__(self, num_nodes: int):
        """Initialize traffic matrix.
        
        Args:
            num_nodes: Number of nodes
        """
        self.num_nodes = num_nodes
        self.matrix = np.zeros((num_nodes, num_nodes))
    
    def set_demand(self, src: int, dst: int, demand: float):
        """Set traffic demand between two nodes.
        
        Args:
            src: Source node
            dst: Destination node
            demand: Demand in Mbps
        """
        if 0 <= src < self.num_nodes and 0 <= dst < self.num_nodes:
            self.matrix[src, dst] = demand
    
    def get_demand(self, src: int, dst: int) -> float:
        """Get traffic demand between two nodes.
        
        Args:
            src: Source node
            dst: Destination node
        
        Returns:
            Demand in Mbps
        """
        if 0 <= src < self.num_nodes and 0 <= dst < self.num_nodes:
            return self.matrix[src, dst]
        return 0.0
    
    def total_demand(self) -> float:
        """Get total demand across all flows.
        
        Returns:
            Total demand in Mbps
        """
        return np.sum(self.matrix)
    
    def max_demand(self) -> float:
        """Get maximum demand across all flows.
        
        Returns:
            Maximum demand in Mbps
        """
        return np.max(self.matrix)
    
    def normalize(self):
        """Normalize traffic matrix to [0, 1] range."""
        max_val = self.max_demand()
        if max_val > 0:
            self.matrix = self.matrix / max_val
    
    def reset(self):
        """Reset traffic matrix to zeros."""
        self.matrix = np.zeros((self.num_nodes, self.num_nodes))
