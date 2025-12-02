"""
Main entry point for GNN-DRL backend application.
Provides CLI interface for inference, training, and API server.
"""

import argparse
import sys
import os
from typing import Optional

# Handle both package and script imports
try:
    from .app_orchestrator import NetworkRoutingSimulator
    from .config import get_config
except ImportError:
    from app_orchestrator import NetworkRoutingSimulator
    from config import get_config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="GNN-DRL Network Routing Optimization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on NSFNet topology with 20 flows
  python main.py --mode inference --topology nsfnet --flows 20
  
  # Train agent for 100 episodes
  python main.py --mode train --episodes 100
  
  # Start API server
  python main.py --mode api --port 8000
  
  # Run simulation with custom configuration
  python main.py --mode simulate --config custom_config.json
        """
    )
    
    # Global arguments
    parser.add_argument(
        '--env',
        type=str,
        default='development',
        choices=['development', 'testing', 'production'],
        help='Configuration environment'
    )
    
    # Mode selection
    subparsers = parser.add_subparsers(dest='mode', help='Operation mode')
    
    # Inference mode
    inference_parser = subparsers.add_parser('inference', help='Run inference')
    inference_parser.add_argument('--topology', default='nsfnet',
                                 choices=['nsfnet', 'geant2', 'random', 'barabasi', 'watts_strogatz'],
                                 help='Network topology type')
    inference_parser.add_argument('--nodes', type=int, default=14,
                                 help='Number of nodes')
    inference_parser.add_argument('--flows', type=int, default=20,
                                 help='Number of flows')
    inference_parser.add_argument('--duration', type=int, default=100,
                                 help='Simulation duration')
    inference_parser.add_argument('--model', type=str, default=None,
                                 help='Path to pre-trained model')
    inference_parser.add_argument('--learning', action='store_true',
                                 help='Enable online learning')
    inference_parser.add_argument('--output', type=str, default='./results/inference',
                                 help='Output directory for results')
    
    # Training mode
    train_parser = subparsers.add_parser('train', help='Train agent')
    train_parser.add_argument('--episodes', type=int, default=500,
                             help='Number of episodes')
    train_parser.add_argument('--topologies', type=int, default=402,
                             help='Number of topologies per episode')
    train_parser.add_argument('--batch-size', type=int, default=32,
                             help='Batch size for learning')
    train_parser.add_argument('--save-interval', type=int, default=10,
                             help='Save checkpoint every N episodes')
    train_parser.add_argument('--output', type=str, default='./models',
                             help='Output directory for models')
    
    # Simulation mode
    sim_parser = subparsers.add_parser('simulate', help='Run custom simulation')
    sim_parser.add_argument('--topology', default='nsfnet',
                           help='Topology file or type')
    sim_parser.add_argument('--flows', type=int, default=20,
                           help='Number of flows')
    sim_parser.add_argument('--config', type=str, default=None,
                           help='Configuration file (JSON)')
    sim_parser.add_argument('--output', type=str, default='./results/simulation',
                           help='Output directory')
    
    # API mode
    api_parser = subparsers.add_parser('api', help='Start REST API server')
    api_parser.add_argument('--host', type=str, default='0.0.0.0',
                           help='Server host')
    api_parser.add_argument('--port', type=int, default=8000,
                           help='Server port')
    api_parser.add_argument('--workers', type=int, default=4,
                           help='Number of workers')
    api_parser.add_argument('--model', type=str, default=None,
                           help='Path to pre-trained model')
    api_parser.add_argument('--debug', action='store_true',
                           help='Enable debug mode')
    
    # Evaluation mode
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model')
    eval_parser.add_argument('--model', type=str, required=True,
                            help='Path to model to evaluate')
    eval_parser.add_argument('--topologies', type=int, default=10,
                            help='Number of topologies to test')
    eval_parser.add_argument('--output', type=str, default='./results/evaluation',
                            help='Output directory')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize simulator
    simulator = NetworkRoutingSimulator(config_env=args.env)
    
    # Get model path if provided
    model_path = getattr(args, 'model', None)
    simulator.initialize(model_path=model_path)
    
    # Handle different modes
    if args.mode == 'inference':
        run_inference(simulator, args)
    elif args.mode == 'train':
        run_training(simulator, args)
    elif args.mode == 'simulate':
        run_simulation(simulator, args)
    elif args.mode == 'api':
        run_api_server(simulator, args)
    elif args.mode == 'evaluate':
        run_evaluation(simulator, args)
    else:
        parser.print_help()
        return 1
    
    return 0


def run_inference(simulator: NetworkRoutingSimulator, args):
    """Run inference mode."""
    print(f"Running inference on {args.topology} topology with {args.flows} flows...")
    
    # Create topology
    simulator.create_topology(args.topology, args.nodes)
    simulator.setup_simulator()
    
    # Generate traffic
    simulator.generate_traffic(num_flows=args.flows)
    
    # Run simulation
    print("Starting simulation...")
    for step in range(args.duration):
        metrics = simulator.run_simulation_step(step)
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}/{args.duration}")
    
    # Print results
    print("\n=== Inference Results ===")
    print(simulator.metrics_tracker.get_episode_summary())
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    metrics_file = os.path.join(args.output, 'metrics.json')
    simulator.metrics_tracker.export_metrics(metrics_file)
    print(f"\nResults saved to {metrics_file}")


def run_training(simulator: NetworkRoutingSimulator, args):
    """Run training mode."""
    print(f"Starting training for {args.episodes} episodes...")
    
    simulator.train(
        num_episodes=args.episodes,
        save_interval=args.save_interval
    )
    
    print("\n=== Training Complete ===")
    print(f"Best model: {simulator.version_manager.get_best_model()}")
    print(f"Total versions: {len(simulator.version_manager.versions)}")


def run_simulation(simulator: NetworkRoutingSimulator, args):
    """Run custom simulation."""
    print("Running custom simulation...")
    
    # Load config if provided
    if args.config:
        print(f"Loading configuration from {args.config}")
        # TODO: Load config from file
    
    # Create topology
    simulator.create_topology(args.topology)
    simulator.setup_simulator()
    
    # Generate traffic
    simulator.generate_traffic(num_flows=args.flows)
    
    # Run episode
    print("Starting simulation...")
    simulator.run_episode(0, learning_enabled=False)
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    metrics_file = os.path.join(args.output, 'metrics.json')
    simulator.metrics_tracker.export_metrics(metrics_file)
    print(f"Results saved to {metrics_file}")


def run_api_server(simulator: NetworkRoutingSimulator, args):
    """Run API server."""
    print(f"Starting API server on {args.host}:{args.port}...")
    
    # Import and run API server
    try:
        try:
            from .api import create_app
        except ImportError:
            from api import create_app
        
        app = create_app(simulator)
        import uvicorn
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError as e:
        print(f"API module not available: {e}")
        print("Please install FastAPI: pip install fastapi uvicorn")
        sys.exit(1)


def run_evaluation(simulator: NetworkRoutingSimulator, args):
    """Run model evaluation."""
    print(f"Evaluating model: {args.model}")
    print(f"Testing on {args.topologies} topologies...")
    
    # Load model
    simulator.gnn_agent.load_checkpoint(args.model)
    
    results = []
    
    for i in range(args.topologies):
        # Create random topology
        simulator.create_topology("random")
        simulator.setup_simulator()
        
        # Generate traffic
        simulator.generate_traffic()
        
        # Run episode
        summary = simulator.run_episode(i, learning_enabled=False)
        results.append(summary)
        
        if (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{args.topologies} topologies")
    
    # Print evaluation results
    print("\n=== Evaluation Results ===")
    avg_reward = sum(r.get('learning', {}).get('episode_reward', 0) 
                    for r in results) / len(results)
    print(f"Average Reward: {avg_reward:.2f}")
    
    # Save results
    os.makedirs(args.output, exist_ok=True)
    results_file = os.path.join(args.output, 'evaluation_results.json')
    import json
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    sys.exit(main())
