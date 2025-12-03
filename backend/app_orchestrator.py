"""
Main application orchestrator for GNN-DRL backend.
Coordinates all components for complete routing optimization workflow.
"""

import numpy as np
import networkx as nx
import torch
from typing import Dict, List, Optional, Tuple, Any

# Handle both package and script imports
try:
    from .config import get_config
    from .topology_manager import NetworkTopologyBuilder, NetworkTopologyValidator
    from .traffic_generator import TrafficDemandGenerator, TrafficMatrix, Flow
    from .network_simulator import NetworkSimulator, LinkState
    from .gnn_agent import create_gnn_agent
    from .inference_engine import GNNInferenceEngine, StatePreprocessor
    from .learning_module import OnlineLearningModule
    from .version_manager import ModelVersionManager
    from .metrics_tracker import MetricsTracker, QoSMetrics, UtilizationMetrics, LearningMetrics, OverheadMetrics
except ImportError:
    from config import get_config
    from topology_manager import NetworkTopologyBuilder, NetworkTopologyValidator
    from traffic_generator import TrafficDemandGenerator, TrafficMatrix, Flow
    from network_simulator import NetworkSimulator, LinkState
    from gnn_agent import create_gnn_agent
    from inference_engine import GNNInferenceEngine, StatePreprocessor
    from learning_module import OnlineLearningModule
    from version_manager import ModelVersionManager
    from metrics_tracker import MetricsTracker, QoSMetrics, UtilizationMetrics, LearningMetrics, OverheadMetrics


class NetworkRoutingSimulator:
    """Main simulator orchestrator combining all components."""
    
    def __init__(self, config_env: str = "development"):
        """Initialize network routing simulator.
        
        Args:
            config_env: Configuration environment (development, testing, production)
        """
        self.config = get_config(config_env)
        
        # Core components
        self.topology_builder = NetworkTopologyBuilder()
        self.traffic_generator = TrafficDemandGenerator(
            self.config.network.num_nodes
        )
        self.simulator: Optional[NetworkSimulator] = None
        self.gnn_agent = None
        self.inference_engine: Optional[GNNInferenceEngine] = None
        self.learning_module: Optional[OnlineLearningModule] = None
        
        # Management
        self.version_manager = ModelVersionManager(
            self.config.gnn.model_checkpoint_dir
        )
        self.metrics_tracker = MetricsTracker()
        
        # State
        self.current_topology: Optional[nx.Graph] = None
        self.flows: List[Flow] = []
        self.flow_routes: Dict[int, List[int]] = {}
        self.simulation_time = 0
    
    def initialize(self, model_path: Optional[str] = None):
        """Initialize all components.
        
        Args:
            model_path: Optional path to pre-trained model
        """
        # Note: If an old checkpoint is provided, we use a fresh model
        # because the old format is incompatible with the current PyTorch architecture
        
        # Create GNN agent
        self.gnn_agent = create_gnn_agent(
            node_dim=self.config.gnn.node_feature_dim,
            hidden_dim=self.config.gnn.hidden_dim,
            num_layers=self.config.gnn.num_gnn_layers,
            num_actions=self.config.gnn.num_classes,
            learning_rate=self.config.gnn.learning_rate
        )
        
        # Create inference engine
        self.inference_engine = GNNInferenceEngine(self.gnn_agent)
        
        # Load pre-trained model if provided
        if model_path:
            import os
            if os.path.exists(model_path):
                print(f"Loading model from {model_path}...")
                try:
                    self.gnn_agent.load_checkpoint(model_path)
                except Exception as e:
                    print(f"Warning: Could not load checkpoint: {e}")
                    print("Using fresh model with random initialization.")
            else:
                print(f"Model file not found: {model_path}")
        
        # Create learning module
        self.learning_module = OnlineLearningModule(
            self.gnn_agent,
            initial_epsilon=self.config.training.epsilon_start,
            epsilon_decay=self.config.training.epsilon_decay,
            epsilon_min=self.config.training.epsilon_end,
            gamma=self.config.training.gamma
        )
    
    def create_topology(self, topology_type: str = "nsfnet",
                       num_nodes: int = 14) -> nx.Graph:
        """Create network topology.
        
        Args:
            topology_type: Type of topology (nsfnet, geant2, random)
            num_nodes: Number of nodes
        
        Returns:
            Created topology
        """
        if topology_type == "nsfnet":
            self.current_topology = self.topology_builder.build_nsfnet(num_nodes)
        elif topology_type == "geant2":
            self.current_topology = self.topology_builder.build_geant2(num_nodes)
        elif topology_type == "random":
            self.current_topology = self.topology_builder.build_random(
                num_nodes, edge_probability=0.3
            )
        else:
            raise ValueError(f"Unknown topology type: {topology_type}. Supported types: nsfnet, geant2, random")
        
        # Add link weights
        self.topology_builder.add_link_weights(
            weight_type="random",
            capacity=self.config.network.link_capacity
        )
        
        # Validate topology
        is_valid, errors = NetworkTopologyValidator.validate(self.current_topology)
        if not is_valid:
            raise ValueError(f"Invalid topology: {errors}")
        
        return self.current_topology
    
    def setup_simulator(self, topology: nx.Graph = None):
        """Setup network simulator.
        
        Args:
            topology: Topology to simulate on (uses current if None)
        """
        if topology is None:
            topology = self.current_topology
        
        if topology is None:
            raise ValueError("No topology specified or created")
        
        self.simulator = NetworkSimulator(
            topology,
            link_capacity=self.config.network.link_capacity
        )
        
        # Setup inference engine
        if self.inference_engine:
            self.inference_engine.setup(topology)
    
    def generate_traffic(self, pattern: str = "uniform",
                        num_flows: int = None) -> List[Flow]:
        """Generate traffic demands.
        
        Args:
            pattern: Traffic pattern type
            num_flows: Number of dynamic flows
        
        Returns:
            List of flows
        """
        if num_flows is None:
            num_flows = self.config.network.traffic_matrix_size
        
        self.flows = self.traffic_generator.generate_dynamic_flows(
            num_flows=num_flows,
            simulation_duration=self.config.network.simulation_duration,
            pattern=pattern
        )
        
        return self.flows
    
    def run_inference_step(self, step: int) -> Dict[int, int]:
        """Run inference for routing decisions.
        
        Args:
            step: Current simulation step
        
        Returns:
            Dictionary of {flow_id: routing_action}
        """
        routing_actions = {}
        
        active_flows = self.traffic_generator.get_active_flows(
            self.flows, step
        )
        
        for flow in active_flows:
            action = self.inference_engine.infer_routing_action(
                self.current_topology,
                self.simulator.links,
                flow.flow_id,
                flow.source,
                flow.destination,
                deterministic=False
            )
            routing_actions[flow.flow_id] = action
        
        return routing_actions
    
    def run_simulation_step(self, step: int) -> Dict[str, Any]:
        """Run one simulation step.
        
        Args:
            step: Current step number
        
        Returns:
            Metrics for this step
        """
        # Get active flows
        active_flows = self.traffic_generator.get_active_flows(self.flows, step)
        
        # Get routing decisions
        routing_actions = self.run_inference_step(step)
        
        # Route flows if not already routed
        for flow in active_flows:
            if flow.flow_id not in self.simulator.flow_routes:
                action = routing_actions.get(flow.flow_id, 0)
                path = self.simulator.route_flow(
                    flow.flow_id,
                    flow.source,
                    flow.destination,
                    action
                )
                self.flow_routes[flow.flow_id] = path
        
        # Forward flows
        flows_to_forward = {f.flow_id: f.data_rate for f in active_flows}
        self.simulator.step(flows_to_forward, routing_actions)
        
        # Compute metrics for this step
        qos_metrics = self._compute_qos_metrics(active_flows)
        util_metrics = self._compute_utilization_metrics()
        overhead_metrics = self._compute_overhead_metrics(routing_actions)
        
        # Record metrics
        for flow_id, metrics in qos_metrics.items():
            self.metrics_tracker.record_qos_metrics(flow_id, metrics)
        self.metrics_tracker.record_utilization(util_metrics)
        self.metrics_tracker.record_overhead(overhead_metrics)
        
        self.simulation_time = step
        
        return {
            'step': step,
            'active_flows': len(active_flows),
            'qos': [m.to_dict() for m in qos_metrics.values()],
            'utilization': util_metrics.to_dict(),
            'overhead': overhead_metrics.to_dict()
        }
    
    def _compute_qos_metrics(self, flows: List[Flow]) -> Dict[int, QoSMetrics]:
        """Compute QoS metrics for flows.
        
        Args:
            flows: List of active flows
        
        Returns:
            Dictionary of QoS metrics by flow
        """
        metrics = {}
        
        for flow in flows:
            if flow.flow_id in self.simulator.flows:
                flow_metrics = self.simulator.flows[flow.flow_id]
                
                qos = QoSMetrics(
                    latency_ms=flow_metrics.average_latency,
                    jitter_ms=5.0,  # Could compute from variance
                    packet_loss_rate=flow_metrics.packet_loss_rate,
                    throughput_mbps=flow_metrics.throughput_mbps,
                    sla_violation_rate=0.0 if flow_metrics.average_latency < 50 else 1.0
                )
                metrics[flow.flow_id] = qos
        
        return metrics
    
    def _compute_utilization_metrics(self) -> UtilizationMetrics:
        """Compute network utilization metrics.
        
        Returns:
            Utilization metrics
        """
        if not self.simulator.links:
            return UtilizationMetrics()
        
        utilizations = [link.utilization for link in self.simulator.links.values()]
        
        #   fairness index
        if utilizations:
            sum_sq = np.sum([u ** 2 for u in utilizations])
            sum_u = np.sum(utilizations)
            n = len(utilizations)
            j = (sum_u ** 2) / (n * sum_sq) if sum_sq > 0 else 1.0
        else:
             j = 1.0
        
        congested = sum(1 for u in utilizations if u > 80)
        
        return UtilizationMetrics(
            avg_link_utilization=np.mean(utilizations) if utilizations else 0.0,
            max_link_utilization=np.max(utilizations) if utilizations else 0.0,
            j_fairness=j,
            load_balance_index=1.0 - (np.std(utilizations) / (np.mean(utilizations) + 1e-6)),
            congested_links=congested
        )
    
    def _compute_overhead_metrics(self, routing_actions: Dict) -> OverheadMetrics:
        """Compute overhead metrics.
        
        Args:
            routing_actions: Routing actions taken
        
        Returns:
            Overhead metrics
        """
        return OverheadMetrics(
            routing_updates=len(routing_actions),
            control_messages=len(routing_actions) * 2,  # Estimate
            computation_time_ms=1.5,  # Estimate
            memory_usage_mb=50.0  # Estimate
        )
    
    def run_episode(self, episode_num: int, learning_enabled: bool = False) -> Dict:
        """Run complete episode.
        
        Args:
            episode_num: Episode number
            learning_enabled: Whether to enable online learning
        
        Returns:
            Episode summary
        """
        self.metrics_tracker.reset_episode()
        
        cumulative_reward = 0.0
        
        for step in range(self.config.network.simulation_duration):
            # Run simulation step
            step_metrics = self.run_simulation_step(step)
            
            # Calculate reward
            util_metrics = step_metrics['utilization']
            reward = self._calculate_reward(util_metrics)
            cumulative_reward += reward
            
            # Online learning
            if learning_enabled and self.learning_module:
                # Record experience
                state = self._get_state()
                next_state = self._get_state()
                
                for flow_id in [f['flow_id'] for f in step_metrics.get('qos', [])]:
                    self.learning_module.add_experience(
                        state, 0, reward, next_state, False
                    )
                
                # Learn from buffer periodically
                if step % 10 == 0:
                    losses = self.learning_module.learn_from_buffer(
                        batch_size=self.config.training.batch_size
                    )
        
        # End episode
        if self.learning_module:
            self.learning_module.end_episode(cumulative_reward)
        
        # Record learning metrics
        learning_metrics = LearningMetrics(
            episode_reward=cumulative_reward,
            cumulative_reward=cumulative_reward,
            epsilon=self.learning_module.epsilon if self.learning_module else 1.0
        )
        self.metrics_tracker.record_learning_metrics(learning_metrics)
        
        return self.metrics_tracker.get_episode_summary()
    
    def _calculate_reward(self, util_metrics: UtilizationMetrics) -> float:
        """Calculate reward based on metrics.
        
        Args:
            util_metrics: Utilization metrics
        
        Returns:
            Reward value
        """
        # Reward for low utilization variance (balanced load)
        fairness = getattr(util_metrics, 'j_fairness', 0.5)
        
        # Penalize congestion
        congested = getattr(util_metrics, 'congested_links', 0)
        
        reward = fairness * 10 - congested
        return float(reward)
    
    def _get_state(self) -> Dict:
        """Get current network state.
        
        Returns:
            State dictionary for GNN input
        """
        if not self.current_topology or not self.simulator:
            return {}
        
        preprocessor = StatePreprocessor(self.config.network.num_nodes)
        
        return {
            'node_features': preprocessor.get_node_features(
                self.current_topology, self.simulator.links
            ),
            'edge_index': preprocessor.get_edge_features(
                self.current_topology, self.simulator.links
            ),
            'flow_state': preprocessor.encode_flow_state(0, 0, 1),
            'adjacency_matrix': preprocessor.get_adjacency_matrix(
                self.current_topology
            )
        }
    
    def _get_improved_state(self) -> Dict:
        """Get improved network state representation (matches notebook).
        
        Returns:
            State dictionary with node features, edge index, and adjacency matrix
        """
        if not self.current_topology or not self.simulator:
            return {
                'node_features': torch.zeros((1, 2)),
                'edge_index': torch.zeros((2, 0), dtype=torch.long),
                'adjacency_matrix': torch.zeros((1, 1))
            }
        
        nodes = sorted(list(self.current_topology.nodes()))
        num_nodes = len(nodes)
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Node features: [load, queue_length] normalized
        node_features = []
        for node in nodes:
            load = sum(self.simulator.links.get((node, n), type('obj', (), {'current_load': 0})).current_load 
                      for n in nodes if (node, n) in self.simulator.links)
            queue = sum(self.simulator.links.get((node, n), type('obj', (), {'queue_length': 0})).queue_length 
                       for n in nodes if (node, n) in self.simulator.links)
            node_features.append([load / 1000.0, queue / 100.0])
        
        # Edge index
        edge_list = []
        for src, dst in self.simulator.links.keys():
            if src in node_to_idx and dst in node_to_idx:
                edge_list.append([node_to_idx[src], node_to_idx[dst]])
        
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous() if edge_list else torch.zeros((2, 0), dtype=torch.long)
        
        # Adjacency matrix
        adj_matrix = torch.zeros((num_nodes, num_nodes))
        for src, dst in self.simulator.links.keys():
            if src in node_to_idx and dst in node_to_idx:
                adj_matrix[node_to_idx[src], node_to_idx[dst]] = 1.0
        
        return {
            'node_features': torch.tensor(node_features, dtype=torch.float32),
            'edge_index': edge_index,
            'adjacency_matrix': adj_matrix
        }
    
    def _encode_flow_state(self, flow: 'Flow', state: Dict) -> np.ndarray:
        """Encode flow state to 6-feature vector (matches notebook).
        
        Args:
            flow: Flow object
            state: Current network state
        
        Returns:
            Flow state array [src_idx, dst_idx, volume, priority, avg_load, congestion]
        """
        nodes = sorted(list(self.current_topology.nodes()))
        
        # Get node indices (normalized)
        src_idx = nodes.index(flow.source) if flow.source in nodes else 0
        dst_idx = nodes.index(flow.destination) if flow.destination in nodes else 0
        src_idx_norm = src_idx / max(len(nodes), 1)
        dst_idx_norm = dst_idx / max(len(nodes), 1)
        
        # Flow volume (normalized to 0-1, assuming max 1000 Mbps)
        volume_norm = flow.data_rate / 1000.0
        
        # Priority (assuming 0-2, normalize to 0-1)
        priority_norm = getattr(flow, 'priority', 1) / 2.0
        
        # Average network load
        loads = [link.current_load for link in self.simulator.links.values()]
        avg_load = np.mean(loads) / 1000.0 if loads else 0.0
        
        # Congestion (fraction of congested links)
        congestion = len([l for l in loads if l > 500]) / len(loads) if loads else 0.0
        
        return np.array([
            src_idx_norm,
            dst_idx_norm,
            volume_norm,
            priority_norm,
            avg_load,
            congestion
        ], dtype=np.float32)
    
    def _calculate_path_delay(self, path: List[int]) -> float:
        """Calculate total delay for a path.
        
        Args:
            path: List of node IDs forming the path
        
        Returns:
            Total delay in milliseconds
        """
        if not path or len(path) < 2:
            return float('inf')
        
        total_delay = 0.0
        for i in range(len(path) - 1):
            link = (path[i], path[i + 1])
            if link in self.simulator.links:
                link_state = self.simulator.links[link]
                # Propagation delay + queue delay
                queue_delay = link_state.queue_length * 0.1  # 0.1ms per packet in queue
                total_delay += link_state.latency + queue_delay
            else:
                return float('inf')
        
        return total_delay
    
    def train(self, num_episodes: int = None, save_interval: int = 10):
        """Train agent using improved DRL approach.
        
        Args:
            num_episodes: Number of episodes to train
            save_interval: Save checkpoint every N episodes
        """
        if num_episodes is None:
            num_episodes = self.config.training.num_episodes
        
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Configuration:")
        print(f"  Batch size: {self.config.training.batch_size}")
        print(f"  Epsilon: {self.config.training.epsilon_start} -> {self.config.training.epsilon_end}")
        print(f"  Gamma: {self.config.training.gamma}")
        print(f"  Update frequency: 10 steps")
        print("-" * 60)
        
        epsilon = self.config.training.epsilon_start
        epsilon_decay = (self.config.training.epsilon_start - self.config.training.epsilon_end) / num_episodes
        
        for episode in range(num_episodes):
            # Create new topology for this episode
            self.create_topology("random")
            self.setup_simulator()
            
            # Generate traffic
            self.generate_traffic()
            
            # Reset metrics
            self.metrics_tracker.reset_episode()
            episode_reward = 0.0
            
            # Run episode with max 100 steps
            max_steps = min(100, self.config.network.simulation_duration)
            
            for step in range(max_steps):
                # Get active flows
                active_flows = self.traffic_generator.get_active_flows(self.flows, step)
                if not active_flows:
                    continue
                
                # Get current state
                state = self._get_improved_state()
                
                # Select actions for each flow
                routing_actions = {}
                for flow in active_flows:
                    # Update flow state
                    flow_state = self._encode_flow_state(flow, state)
                    
                    # Select action with epsilon-greedy
                    action = self.gnn_agent.select_action(
                        state['node_features'],
                        state['edge_index'],
                        flow_state,
                        state.get('adjacency_matrix'),
                        epsilon=epsilon
                    )
                    routing_actions[flow.flow_id] = action
                    
                    # Route flow if not already routed
                    if flow.flow_id not in self.simulator.flow_routes:
                        path = self.simulator.route_flow(
                            flow.flow_id,
                            flow.source,
                            flow.destination,
                            action
                        )
                        self.flow_routes[flow.flow_id] = path
                        
                        # Calculate path delay
                        delay = self._calculate_path_delay(path)
                        
                        # Compute reward using improved method
                        reward = self.simulator.compute_routing_reward(flow.flow_id, path, delay)
                        episode_reward += reward
                        
                        # Get next state
                        next_state = self._get_improved_state()
                        next_flow_state = self._encode_flow_state(flow, next_state)
                        
                        # Store experience
                        done = (step >= max_steps - 1)
                        self.gnn_agent.store_experience(
                            state['node_features'],
                            flow_state,
                            state['edge_index'],
                            action,
                            reward,
                            next_state['node_features'],
                            next_flow_state,
                            next_state['edge_index'],
                            done
                        )
                
                # Forward flows
                flows_to_forward = {f.flow_id: f.data_rate for f in active_flows}
                self.simulator.step(flows_to_forward, routing_actions)
                
                # Update policy every 10 steps
                if (step + 1) % 10 == 0:
                    losses = self.gnn_agent.update_policy(
                        batch_size=self.config.training.batch_size,
                        gamma=self.config.training.gamma
                    )
            
            # Record episode completion
            self.gnn_agent.record_episode(episode_reward)
            
            # Decay epsilon
            epsilon = max(self.config.training.epsilon_end, epsilon - epsilon_decay)
            
            # Compute average reward for last 10 episodes
            recent_rewards = self.gnn_agent.episode_rewards[-10:]
            avg_reward = np.mean(recent_rewards) if recent_rewards else episode_reward
            
            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1:3d}/{num_episodes} | "
                      f"Reward: {episode_reward:7.3f} | "
                      f"Avg (last 10): {avg_reward:7.3f} | "
                      f"Epsilon: {epsilon:.3f}")
            
            # Save checkpoint
            if (episode + 1) % save_interval == 0:
                self._save_checkpoint(episode)
        
        print("-" * 60)
        print("✓ Training completed")
        print(f"  Final average reward: {avg_reward:.3f}")
        if self.gnn_agent.episode_rewards:
            print(f"  Best episode reward: {max(self.gnn_agent.episode_rewards):.3f}")
            print(f"  Total episodes: {len(self.gnn_agent.episode_rewards)}")
    
    def _save_checkpoint(self, episode: int):
        """Save training checkpoint.
        
        Args:
            episode: Current episode
        """
        if self.gnn_agent and self.learning_module:
            import os
            
            # Create models directory if it doesn't exist
            models_dir = os.path.dirname(self.config.gnn.latest_model_path)
            os.makedirs(models_dir, exist_ok=True)
            
            # Save the model first
            self.gnn_agent.save_checkpoint(self.config.gnn.latest_model_path)
            
            # Get metrics and create version
            metrics = self.metrics_tracker.compute_aggregate_metrics()
            version_id = self.version_manager.create_version(
                self.config.gnn.latest_model_path,
                metrics,
                notes=f"Episode {episode}"
            )
            print(f"Saved checkpoint: {version_id}")
    
    def get_status(self) -> Dict:
        """Get current simulator status.
        
        Returns:
            Status dictionary
        """
        return {
            'topology': f"{self.config.network.num_nodes} nodes" if self.current_topology else "None",
            'simulation_time': self.simulation_time,
            'active_flows': len(self.flows),
            'learning_stats': self.learning_module.get_learning_statistics() if self.learning_module else {},
            'latest_metrics': self.metrics_tracker.compute_aggregate_metrics()
        }
