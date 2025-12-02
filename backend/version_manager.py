"""
Model version management and checkpointing.
Tracks and manages different model versions.
"""

import os
import json
import shutil
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ModelVersion:
    """Represents a model version."""
    version_id: str
    created_at: str
    metrics: Dict
    performance_score: float
    model_path: str
    is_best: bool = False
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ModelVersionManager:
    """Manages model versions and checkpointing."""
    
    def __init__(self, checkpoint_dir: str = "./models/checkpoints"):
        """Initialize version manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.versions_file = os.path.join(checkpoint_dir, "versions.json")
        self.versions: Dict[str, ModelVersion] = {}
        self.best_version_id: Optional[str] = None
        self.latest_version_id: Optional[str] = None
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        self._load_versions()
    
    def _load_versions(self):
        """Load versions from file."""
        if os.path.exists(self.versions_file):
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                    for version_id, version_data in data.get('versions', {}).items():
                        self.versions[version_id] = ModelVersion(**version_data)
                    self.best_version_id = data.get('best_version_id')
                    self.latest_version_id = data.get('latest_version_id')
            except Exception as e:
                print(f"Error loading versions: {e}")
    
    def _save_versions(self):
        """Save versions to file."""
        data = {
            'versions': {vid: v.to_dict() for vid, v in self.versions.items()},
            'best_version_id': self.best_version_id,
            'latest_version_id': self.latest_version_id,
            'last_updated': datetime.now().isoformat()
        }
        
        os.makedirs(os.path.dirname(self.versions_file), exist_ok=True)
        with open(self.versions_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def create_version(self, model_path: str, metrics: Dict,
                      notes: str = "") -> str:
        """Create a new model version.
        
        Args:
            model_path: Path to model checkpoint
            metrics: Performance metrics
            notes: Optional notes about version
        
        Returns:
            Version ID
        """
        # Generate version ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v_{timestamp}"
        
        # Copy model to versioned location
        version_model_path = os.path.join(self.checkpoint_dir, f"{version_id}_model.pt")
        shutil.copy(model_path, version_model_path)
        
        # Calculate performance score
        performance_score = self._calculate_performance_score(metrics)
        
        # Create version
        version = ModelVersion(
            version_id=version_id,
            created_at=datetime.now().isoformat(),
            metrics=metrics,
            performance_score=performance_score,
            model_path=version_model_path,
            notes=notes
        )
        
        self.versions[version_id] = version
        self.latest_version_id = version_id
        
        # Update best version if this is better
        if (self.best_version_id is None or 
            performance_score > self.versions[self.best_version_id].performance_score):
            if self.best_version_id is not None:
                self.versions[self.best_version_id].is_best = False
            self.versions[version_id].is_best = True
            self.best_version_id = version_id
        
        self._save_versions()
        return version_id
    
    def _calculate_performance_score(self, metrics: Dict) -> float:
        """Calculate composite performance score.
        
        Args:
            metrics: Dictionary of metrics
        
        Returns:
            Performance score
        """
        score = 0.0
        
        # Average reward (higher is better)
        if 'avg_reward' in metrics:
            score += min(metrics['avg_reward'], 1.0) * 0.4
        
        # Success rate (higher is better)
        if 'success_rate' in metrics:
            score += metrics['success_rate'] * 0.3
        
        # QoS metrics (lower latency is better)
        if 'avg_latency_ms' in metrics:
            score += max(0, 1 - min(metrics['avg_latency_ms'] / 100, 1.0)) * 0.15
        
        # Throughput (higher is better)
        if 'avg_throughput' in metrics:
            score += min(metrics['avg_throughput'] / 100, 1.0) * 0.15
        
        return score
    
    def get_best_model(self) -> Optional[str]:
        """Get path to best model.
        
        Returns:
            Path to best model or None
        """
        if self.best_version_id:
            return self.versions[self.best_version_id].model_path
        return None
    
    def get_latest_model(self) -> Optional[str]:
        """Get path to latest model.
        
        Returns:
            Path to latest model or None
        """
        if self.latest_version_id:
            return self.versions[self.latest_version_id].model_path
        return None
    
    def get_version(self, version_id: str) -> Optional[ModelVersion]:
        """Get version by ID.
        
        Args:
            version_id: Version identifier
        
        Returns:
            ModelVersion or None
        """
        return self.versions.get(version_id)
    
    def list_versions(self, limit: int = 10) -> List[ModelVersion]:
        """List versions sorted by creation time.
        
        Args:
            limit: Maximum number of versions to return
        
        Returns:
            List of ModelVersion objects
        """
        sorted_versions = sorted(
            self.versions.values(),
            key=lambda v: v.created_at,
            reverse=True
        )
        return sorted_versions[:limit]
    
    def get_version_history(self) -> Dict[str, float]:
        """Get performance history across versions.
        
        Returns:
            Dictionary of {version_id: performance_score}
        """
        return {
            vid: version.performance_score
            for vid, version in self.versions.items()
        }
    
    def delete_version(self, version_id: str):
        """Delete a model version.
        
        Args:
            version_id: Version to delete
        """
        if version_id in self.versions:
            version = self.versions[version_id]
            
            # Delete model file
            if os.path.exists(version.model_path):
                os.remove(version.model_path)
            
            # Remove from tracking
            del self.versions[version_id]
            
            # Update best/latest if needed
            if self.best_version_id == version_id:
                self.best_version_id = None
            if self.latest_version_id == version_id:
                self.latest_version_id = None
            
            self._save_versions()
    
    def cleanup_old_versions(self, keep_count: int = 5):
        """Remove old model versions keeping only recent ones.
        
        Args:
            keep_count: Number of versions to keep
        """
        sorted_versions = sorted(
            self.versions.items(),
            key=lambda x: x[1].created_at,
            reverse=True
        )
        
        to_delete = sorted_versions[keep_count:]
        for version_id, _ in to_delete:
            self.delete_version(version_id)
    
    def export_version(self, version_id: str, export_path: str):
        """Export a version with metadata.
        
        Args:
            version_id: Version to export
            export_path: Path to export to
        """
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        
        # Create export directory
        os.makedirs(export_path, exist_ok=True)
        
        # Copy model
        shutil.copy(
            version.model_path,
            os.path.join(export_path, "model.pt")
        )
        
        # Save metadata
        metadata = version.to_dict()
        metadata['model_path'] = 'model.pt'  # Relative path
        
        with open(os.path.join(export_path, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)


class CheckpointManager:
    """Manages training checkpoints."""
    
    def __init__(self, checkpoint_dir: str = "./models/checkpoints"):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    def save_training_checkpoint(self, episode: int, agent_state: Dict,
                                learning_state: Dict):
        """Save training checkpoint.
        
        Args:
            episode: Current episode number
            agent_state: Agent state dict
            learning_state: Learning state dict
        """
        checkpoint = {
            'episode': episode,
            'timestamp': datetime.now().isoformat(),
            'agent_state': agent_state,
            'learning_state': learning_state
        }
        
        path = os.path.join(self.checkpoint_dir, f"checkpoint_ep{episode}.json")
        with open(path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def load_training_checkpoint(self, episode: int) -> Optional[Dict]:
        """Load training checkpoint.
        
        Args:
            episode: Episode number
        
        Returns:
            Checkpoint data or None
        """
        path = os.path.join(self.checkpoint_dir, f"checkpoint_ep{episode}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None
