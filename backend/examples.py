"""
Example usage scripts for GNN-DRL backend.
Demonstrates common workflows and patterns.
"""

# Handle both package and script imports
def _import_module(module_name):
    """Import module handling both package and script contexts."""
    try:
        # Try package import first
        return __import__(f'backend.{module_name}', fromlist=[module_name.split('.')[-1]])
    except ImportError:
        # Fall back to direct import
        return __import__(module_name)


# Example 1: Basic Inference
def example_basic_inference():
    """Example: Run inference on a network topology."""
    app_orchestrator = _import_module('app_orchestrator')
    NetworkRoutingSimulator = app_orchestrator.NetworkRoutingSimulator
    
    # Initialize
    simulator = NetworkRoutingSimulator(config_env='development')
    simulator.initialize()
    
    # Create topology
    simulator.create_topology(topology_type='nsfnet', num_nodes=14)
    simulator.setup_simulator()
    
    # Generate traffic
    simulator.generate_traffic(pattern='uniform', num_flows=20)
    
    # Run simulation
    print("Running inference simulation...")
    for step in range(100):
        metrics = simulator.run_simulation_step(step)
        
        if (step + 1) % 25 == 0:
            util = metrics['utilization']
            print(f"Step {step + 1}: Avg Link Util: {util['avg_link_utilization']:.1f}%")
    
    # Print summary
    summary = simulator.metrics_tracker.get_episode_summary()
    print("\n=== Results ===")
    
    # Check if QoS metrics are available
    if summary.get('qos') and isinstance(summary['qos'], dict) and summary['qos']:
        print(f"Average Latency: {summary['qos'].get('avg_latency_ms', 0):.2f} ms")
        print(f"Packet Loss: {summary['qos'].get('avg_packet_loss', 0):.4f}")
    else:
        print("QoS metrics: Not available (no completed flows)")
    
    print(f"j Fairness: {summary['utilization']['j_fairness']:.3f}")
    print(f"Avg Link Utilization: {summary['utilization']['avg_link_utilization']:.2f}%")


# Example 2: Training with Online Learning
def example_training():
    """Example: Train agent with online learning."""
    app_orchestrator = _import_module('app_orchestrator')
    NetworkRoutingSimulator = app_orchestrator.NetworkRoutingSimulator
    
    simulator = NetworkRoutingSimulator(config_env='development')
    simulator.initialize()
    
    # Train for a few episodes
    simulator.train(num_episodes=10, save_interval=5)
    
    # Check results
    print(f"Best model path: {simulator.version_manager.get_best_model()}")
    print(f"Total versions: {len(simulator.version_manager.versions)}")
    
    # List versions
    for version in simulator.version_manager.list_versions():
        print(f"  {version.version_id}: score={version.performance_score:.3f}")


# Example 3: Evaluation on Multiple Topologies
def example_evaluation():
    """Example: Evaluate trained model on various topologies."""
    app_orchestrator = _import_module('app_orchestrator')
    NetworkRoutingSimulator = app_orchestrator.NetworkRoutingSimulator
    
    simulator = NetworkRoutingSimulator(config_env='production')
    simulator.initialize(model_path='./models/best_model.pt')
    
    results = []
    topologies = ['nsfnet', 'geant2', 'random', 'barabasi']
    
    for topology_type in topologies:
        print(f"\nEvaluating on {topology_type}...")
        
        # Create topology
        simulator.create_topology(topology_type=topology_type)
        simulator.setup_simulator()
        
        # Generate traffic
        simulator.generate_traffic(pattern='uniform', num_flows=15)
        
        # Run episode
        summary = simulator.run_episode(0, learning_enabled=False)
        results.append({
            'topology': topology_type,
            'metrics': summary
        })
    
    # Print results
    print("\n=== Evaluation Results ===")
    for result in results:
        qos = result['metrics'].get('qos', {})
        util = result['metrics'].get('utilization', {})
        if qos and 'avg_latency_ms' in qos:
            print(f"{result['topology']}: Latency={qos['avg_latency_ms']:.1f}ms, Loss={qos.get('avg_packet_loss', 0):.4f}")
        else:
            print(f"{result['topology']}: Fairness={util.get('j_fairness', 0):.3f}, Util={util.get('avg_link_utilization', 0):.1f}%")


# Example 4: Custom Topology and Traffic
def example_custom_setup():
    """Example: Use custom topology configuration."""
    import networkx as nx
    network_simulator = _import_module('network_simulator')
    traffic_generator = _import_module('traffic_generator')
    NetworkSimulator = network_simulator.NetworkSimulator
    TrafficDemandGenerator = traffic_generator.TrafficDemandGenerator
    
    # Create custom topology
    graph = nx.path_graph(6)  # Linear topology
    
    # Add link weights
    for u, v in graph.edges():
        graph[u][v]['capacity'] = 100.0
        graph[u][v]['latency'] = 5.0 + u * 0.5
    
    # Create simulator
    simulator = NetworkSimulator(graph, link_capacity=100.0)
    
    # Generate traffic
    traffic_gen = TrafficDemandGenerator(num_nodes=6)
    flows = traffic_gen.generate_dynamic_flows(
        num_flows=10,
        simulation_duration=50,
        pattern='poisson',
        arrival_rate=0.5
    )
    
    # Run simulation
    for step in range(50):
        active_flows = traffic_gen.get_active_flows(flows, step)
        print(f"Step {step}: {len(active_flows)} active flows")


# Example 5: Model Management
def example_model_management():
    """Example: Manage model versions."""
    version_manager = _import_module('version_manager')
    ModelVersionManager = version_manager.ModelVersionManager
    
    manager = ModelVersionManager(checkpoint_dir='./models/checkpoints')
    
    # List versions
    print("Available versions:")
    for version in manager.list_versions(limit=5):
        print(f"  {version.version_id}")
        print(f"    Score: {version.performance_score:.3f}")
        print(f"    Created: {version.created_at}")
    
    # Get best model
    best_path = manager.get_best_model()
    print(f"\nBest model: {best_path}")
    
    # Export version
    best_id = manager.best_version_id
    if best_id:
        manager.export_version(best_id, './models/export/')
        print(f"Exported {best_id} to ./models/export/")
    
    # Cleanup old versions
    print("\nCleaning up old versions...")
    manager.cleanup_old_versions(keep_count=5)


# Example 6: Metrics Analysis
def example_metrics_analysis():
    """Example: Analyze collected metrics."""
    metrics_tracker = _import_module('metrics_tracker')
    MetricsTracker = metrics_tracker.MetricsTracker
    QoSMetrics = metrics_tracker.QoSMetrics
    UtilizationMetrics = metrics_tracker.UtilizationMetrics
    
    tracker = MetricsTracker(window_size=100)
    
    # Simulate some metrics collection
    for step in range(50):
        # Record QoS metrics
        qos = QoSMetrics(
            latency_ms=45 + step * 0.1,
            jitter_ms=2 + step * 0.01,
            packet_loss_rate=0.001,
            throughput_mbps=85 + step * 0.2,
            sla_violation_rate=0.0
        )
        tracker.record_qos_metrics(flow_id=0, metrics=qos)
        
        # Record utilization
        util = UtilizationMetrics(
            avg_link_utilization=50 + step * 0.5,
            max_link_utilization=85 + step * 0.3,
            j_fairness=0.88,
            congested_links=2 if step > 20 else 0
        )
        tracker.record_utilization(util)
    
    # Get summary
    summary = tracker.get_episode_summary()
    print("Metrics Summary:")
    print(f"Average Latency: {summary['qos']['avg_latency_ms']:.2f} ms")
    print(f"Max Latency: {summary['qos']['max_latency_ms']:.2f} ms")
    print(f"j Fairness: {summary['utilization']['j_fairness']:.3f}")
    print(f"Congested Links: {summary['utilization']['congested_links']}")


# Example 7: Online Learning Integration
def example_online_learning():
    """Example: Use online learning during inference."""
    app_orchestrator = _import_module('app_orchestrator')
    learning_module = _import_module('learning_module')
    NetworkRoutingSimulator = app_orchestrator.NetworkRoutingSimulator
    OnlineLearningModule = learning_module.OnlineLearningModule
    
    simulator = NetworkRoutingSimulator(config_env='development')
    simulator.initialize()
    
    # Create topology
    simulator.create_topology(topology_type='random', num_nodes=10)
    simulator.setup_simulator()
    
    # Generate traffic
    simulator.generate_traffic(num_flows=15)
    
    # Run with online learning
    print("Running with online learning enabled...")
    summary = simulator.run_episode(0, learning_enabled=True)
    
    # Check learning statistics
    stats = simulator.learning_module.get_learning_statistics()
    print(f"\nLearning Statistics:")
    print(f"  Total Updates: {stats['total_updates']}")
    print(f"  Buffer Size: {stats['buffer_size']}")
    print(f"  Epsilon: {stats['epsilon']:.4f}")


# Example 8: Configuration Management
def example_configuration():
    """Example: Work with configurations."""
    config_module = _import_module('config')
    get_config = config_module.get_config
    Config = config_module.Config
    
    # Get default configuration
    config = get_config(env='development')
    
    # Access sub-configurations
    print("Network Configuration:")
    print(f"  Nodes: {config.network.num_nodes}")
    print(f"  Link Capacity: {config.network.link_capacity} Mbps")
    print(f"  Simulation Duration: {config.network.simulation_duration} steps")
    
    print("\nGNN Configuration:")
    print(f"  Hidden Dim: {config.gnn.hidden_dim}")
    print(f"  Learning Rate: {config.gnn.learning_rate}")
    print(f"  Num Layers: {config.gnn.num_gnn_layers}")
    
    print("\nTraining Configuration:")
    print(f"  Episodes: {config.training.num_episodes}")
    print(f"  Batch Size: {config.training.batch_size}")
    
    # Convert to dictionary
    config_dict = config.to_dict()
    print(f"\nFull config keys: {list(config_dict.keys())}")


# Example 9: Inference Engine Usage
def example_inference_engine():
    """Example: Use inference engine directly."""
    inference_engine = _import_module('inference_engine')
    topology_manager = _import_module('topology_manager')
    GNNInferenceEngine = inference_engine.GNNInferenceEngine
    NetworkTopologyBuilder = topology_manager.NetworkTopologyBuilder
    import networkx as nx
    
    # Build topology
    builder = NetworkTopologyBuilder()
    topology = builder.build_nsfnet(num_nodes=14)
    
    # Create inference engine
    engine = GNNInferenceEngine()
    engine.setup(topology)
    
    # Get dummy link states
    link_states = {(u, v): type('LinkState', (), {
        'utilization': 50.0
    })() for u, v in topology.edges()}
    
    # Run inference for some flows
    flows = [
        (0, 0, 5),   # flow_id, source, destination
        (1, 2, 8),
        (2, 4, 10),
    ]
    
    actions = engine.infer_batch(topology, link_states, flows, deterministic=True)
    print(f"Routing actions: {actions}")


if __name__ == '__main__':
    import sys
    
    examples = {
        '1': ('Basic Inference', example_basic_inference),
        '2': ('Training', example_training),
        '3': ('Evaluation', example_evaluation),
        '4': ('Custom Setup', example_custom_setup),
        '5': ('Model Management', example_model_management),
        '6': ('Metrics Analysis', example_metrics_analysis),
        '7': ('Online Learning', example_online_learning),
        '8': ('Configuration', example_configuration),
        '9': ('Inference Engine', example_inference_engine),
    }
    
    print("GNN-DRL Backend Examples")
    print("=" * 40)
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")
    print("=" * 40)
    
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        if choice in examples:
            print(f"\nRunning: {examples[choice][0]}\n")
            examples[choice][1]()
        else:
            print(f"Invalid choice: {choice}")
    else:
        print("\nUsage: python examples.py <number>")
        print("Example: python examples.py 1")
