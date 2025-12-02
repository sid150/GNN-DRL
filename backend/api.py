"""
REST API module for GNN-DRL backend.
Provides FastAPI endpoints for inference, training, and model management.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
from datetime import datetime


# Data Models
class TopologyRequest(BaseModel):
    """Request for topology creation."""
    topology_type: str = "nsfnet"
    num_nodes: int = 14


class InferenceRequest(BaseModel):
    """Request for inference."""
    flow_id: int
    source: int
    destination: int
    deterministic: bool = False


class TrafficRequest(BaseModel):
    """Request for traffic generation."""
    pattern: str = "uniform"
    num_flows: int = 20
    duration: int = 100


class SimulationRequest(BaseModel):
    """Request for simulation."""
    topology_type: str = "nsfnet"
    pattern: str = "uniform"
    num_flows: int = 20
    duration: int = 100


class ModelVersion(BaseModel):
    """Model version information."""
    version_id: str
    created_at: str
    performance_score: float
    is_best: bool
    model_path: str


class MetricsResponse(BaseModel):
    """Metrics response."""
    timestamp: str
    qos: Dict[str, float]
    utilization: Dict[str, float]
    learning: Dict[str, float]
    overhead: Dict[str, float]


def create_app(simulator=None) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        simulator: NetworkRoutingSimulator instance
    
    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="GNN-DRL Backend API",
        description="Network Routing Optimization API",
        version="1.0.0"
    )
    
    # Store simulator in app state
    if simulator is None:
        from .app_orchestrator import NetworkRoutingSimulator
        simulator = NetworkRoutingSimulator(config_env='production')
        simulator.initialize()
    
    app.state.simulator = simulator
    
    # Health check
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    
    # Topology endpoints
    @app.post("/api/v1/topology/create")
    async def create_topology(request: TopologyRequest):
        """Create network topology.
        
        Args:
            request: Topology creation request
        
        Returns:
            Topology information
        """
        try:
            topology = app.state.simulator.create_topology(
                topology_type=request.topology_type,
                num_nodes=request.num_nodes
            )
            
            return {
                "status": "success",
                "topology": {
                    "type": request.topology_type,
                    "num_nodes": topology.number_of_nodes(),
                    "num_edges": topology.number_of_edges()
                }
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/api/v1/topology/info")
    async def get_topology_info():
        """Get current topology information."""
        if app.state.simulator.current_topology is None:
            raise HTTPException(status_code=404, detail="No topology created")
        
        metadata = app.state.simulator.topology_builder.get_metadata()
        return {
            "name": metadata.name,
            "num_nodes": metadata.num_nodes,
            "num_edges": metadata.num_edges,
            "average_degree": metadata.average_degree,
            "density": metadata.density,
            "diameter": metadata.diameter,
            "is_connected": metadata.is_connected
        }
    
    # Traffic endpoints
    @app.post("/api/v1/traffic/generate")
    async def generate_traffic(request: TrafficRequest):
        """Generate traffic demands.
        
        Args:
            request: Traffic generation request
        
        Returns:
            Generated flows information
        """
        try:
            flows = app.state.simulator.generate_traffic(
                pattern=request.pattern,
                num_flows=request.num_flows
            )
            
            return {
                "status": "success",
                "num_flows": len(flows),
                "pattern": request.pattern,
                "flows": [
                    {
                        "flow_id": f.flow_id,
                        "source": f.source,
                        "destination": f.destination,
                        "data_rate": f.data_rate,
                        "duration": f.duration
                    }
                    for f in flows[:10]  # Return first 10 for brevity
                ]
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # Inference endpoints
    @app.post("/api/v1/inference")
    async def run_inference(request: InferenceRequest):
        """Run inference for routing decision.
        
        Args:
            request: Inference request
        
        Returns:
            Routing action
        """
        try:
            action = app.state.simulator.inference_engine.infer_routing_action(
                app.state.simulator.current_topology,
                app.state.simulator.simulator.links,
                request.flow_id,
                request.source,
                request.destination,
                deterministic=request.deterministic
            )
            
            action_names = ["shortest_hop", "min_delay", "max_capacity"]
            
            return {
                "status": "success",
                "action": action,
                "action_name": action_names[action],
                "flow_id": request.flow_id
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/api/v1/simulation/step")
    async def run_simulation_step(step: int = 0):
        """Run one simulation step.
        
        Args:
            step: Step number
        
        Returns:
            Step metrics
        """
        try:
            if app.state.simulator.simulator is None:
                app.state.simulator.setup_simulator()
            
            metrics = app.state.simulator.run_simulation_step(step)
            return {
                "status": "success",
                "step": step,
                "metrics": metrics
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # Metrics endpoints
    @app.get("/api/v1/metrics/current")
    async def get_current_metrics():
        """Get current metrics."""
        summary = app.state.simulator.metrics_tracker.get_episode_summary()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "qos": summary.get('qos', {}),
            "utilization": summary.get('utilization', {}),
            "learning": summary.get('learning', {}),
            "overhead": summary.get('overhead', {})
        }
    
    @app.get("/api/v1/metrics/export")
    async def export_metrics(format: str = "json"):
        """Export metrics to file.
        
        Args:
            format: Export format (json, csv)
        
        Returns:
            Metrics file path
        """
        try:
            os.makedirs("./results", exist_ok=True)
            filepath = f"./results/metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            app.state.simulator.metrics_tracker.export_metrics(filepath)
            
            return {
                "status": "success",
                "filepath": filepath
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # Model endpoints
    @app.get("/api/v1/models")
    async def list_models(limit: int = 10):
        """List available model versions.
        
        Args:
            limit: Maximum number of versions to return
        
        Returns:
            List of model versions
        """
        versions = app.state.simulator.version_manager.list_versions(limit)
        
        return {
            "status": "success",
            "total": len(app.state.simulator.version_manager.versions),
            "versions": [
                {
                    "version_id": v.version_id,
                    "created_at": v.created_at,
                    "performance_score": v.performance_score,
                    "is_best": v.is_best,
                    "notes": v.notes
                }
                for v in versions
            ]
        }
    
    @app.get("/api/v1/models/best")
    async def get_best_model():
        """Get best model path."""
        best_path = app.state.simulator.version_manager.get_best_model()
        
        if best_path is None:
            raise HTTPException(status_code=404, detail="No model available")
        
        return {
            "status": "success",
            "best_model": best_path,
            "version_id": app.state.simulator.version_manager.best_version_id
        }
    
    @app.post("/api/v1/models/upload")
    async def upload_model(file: UploadFile = File(...), notes: str = ""):
        """Upload new model.
        
        Args:
            file: Model file
            notes: Optional notes
        
        Returns:
            Version ID
        """
        try:
            # Save uploaded file
            os.makedirs("./models/uploads", exist_ok=True)
            filepath = f"./models/uploads/{file.filename}"
            
            with open(filepath, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Create version
            metrics = app.state.simulator.metrics_tracker.compute_aggregate_metrics()
            version_id = app.state.simulator.version_manager.create_version(
                filepath, metrics, notes=notes
            )
            
            return {
                "status": "success",
                "version_id": version_id,
                "model_path": filepath
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.delete("/api/v1/models/{version_id}")
    async def delete_model(version_id: str):
        """Delete model version.
        
        Args:
            version_id: Version to delete
        
        Returns:
            Deletion status
        """
        try:
            app.state.simulator.version_manager.delete_version(version_id)
            
            return {
                "status": "success",
                "message": f"Deleted version {version_id}"
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # Status endpoint
    @app.get("/api/v1/status")
    async def get_status():
        """Get simulator status."""
        return {
            "status": "success",
            "simulator_status": app.state.simulator.get_status()
        }
    
    return app


if __name__ == "__main__":
    import uvicorn
    
    app = create_app()
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        workers=4
    )
