"""
Online learning module for continuous model improvement.
Manages experience buffer and policy updates.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import deque
import json
import os


@dataclass
class Experience:
    """Single experience (state, action, reward, next_state, done)."""
    state: Dict
    action: int
    reward: float
    next_state: Dict
    done: bool


class ExperienceBuffer:
    """Experience replay buffer for learning."""
    
    def __init__(self, capacity: int = 10000):
        """Initialize experience buffer.
        
        Args:
            capacity: Maximum buffer capacity
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def add(self, experience: Experience):
        """Add experience to buffer.
        
        Args:
            experience: Experience to add
        """
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> List[Experience]:
        """Sample batch from buffer.
        
        Args:
            batch_size: Size of batch to sample
        
        Returns:
            List of sampled experiences
        """
        if len(self.buffer) < batch_size:
            return list(self.buffer)
        
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        """Get buffer size."""
        return len(self.buffer)
    
    def clear(self):
        """Clear buffer."""
        self.buffer.clear()


class OnlineLearningModule:
    """Handles online learning and continuous improvement."""
    
    def __init__(self, agent, initial_epsilon: float = 1.0,
                 epsilon_decay: float = 0.995, epsilon_min: float = 0.01,
                 gamma: float = 0.99):
        """Initialize online learning module.
        
        Args:
            agent: GNN agent to train
            initial_epsilon: Initial exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
            gamma: Discount factor
        """
        self.agent = agent
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        
        self.experience_buffer = ExperienceBuffer()
        self.episode_counter = 0
        self.update_counter = 0
        
        # Statistics
        self.episode_rewards = []
        self.loss_history = {
            'policy_loss': [],
            'value_loss': []
        }
    
    def select_action_with_exploration(self, actions_logits: np.ndarray) -> int:
        """Select action using epsilon-greedy strategy.
        
        Args:
            actions_logits: Logits from agent
        
        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(len(actions_logits))
        else:
            # Exploit: best action
            return np.argmax(actions_logits)
    
    def add_experience(self, state: Dict, action: int, reward: float,
                      next_state: Dict, done: bool):
        """Add experience to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        experience = Experience(state, action, reward, next_state, done)
        self.experience_buffer.add(experience)
    
    def learn_from_buffer(self, batch_size: int = 32) -> Dict[str, float]:
        """Learn from experiences in buffer.
        
        Args:
            batch_size: Batch size for learning
        
        Returns:
            Dictionary of loss values
        """
        if len(self.experience_buffer) < batch_size:
            return {'policy_loss': 0.0, 'value_loss': 0.0}
        
        # Sample batch
        batch = self.experience_buffer.sample(batch_size)
        
        # Organize batch
        states = [exp.state for exp in batch]
        actions = np.array([exp.action for exp in batch])
        rewards = np.array([exp.reward for exp in batch])
        next_states = [exp.next_state for exp in batch]
        dones = np.array([exp.done for exp in batch])
        
        # Update agent
        losses = self.agent.update_policy(states, actions, rewards, 
                                         next_states, dones, 
                                         gamma=self.gamma)
        
        # Update counters
        self.update_counter += 1
        
        # Decay epsilon
        self._decay_epsilon()
        
        return losses
    
    def _decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def end_episode(self, episode_reward: float):
        """Called at end of episode.
        
        Args:
            episode_reward: Total reward for episode
        """
        self.episode_counter += 1
        self.episode_rewards.append(episode_reward)
    
    def get_learning_statistics(self) -> Dict:
        """Get learning statistics.
        
        Returns:
            Dictionary of statistics
        """
        if not self.episode_rewards:
            return {}
        
        rewards = np.array(self.episode_rewards)
        
        return {
            'total_episodes': self.episode_counter,
            'total_updates': self.update_counter,
            'avg_reward': np.mean(rewards[-100:]),  # Last 100 episodes
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'epsilon': self.epsilon,
            'buffer_size': len(self.experience_buffer)
        }
    
    def save_checkpoint(self, filepath: str):
        """Save learning state checkpoint.
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'episode_counter': self.episode_counter,
            'update_counter': self.update_counter,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'loss_history': self.loss_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        # Also save agent
        self.agent.save_checkpoint(filepath.replace('.json', '_agent.pt'))
    
    def load_checkpoint(self, filepath: str):
        """Load learning state checkpoint.
        
        Args:
            filepath: Path to load checkpoint from
        """
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.episode_counter = checkpoint.get('episode_counter', 0)
        self.update_counter = checkpoint.get('update_counter', 0)
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.loss_history = checkpoint.get('loss_history', self.loss_history)
        
        # Load agent
        self.agent.load_checkpoint(filepath.replace('.json', '_agent.pt'))


class ContinuousLearningScheduler:
    """Manages continuous learning schedule during inference."""
    
    def __init__(self, learn_rate: int = 10, batch_size: int = 32,
                 update_frequency: int = 100):
        """Initialize learning scheduler.
        
        Args:
            learn_rate: Learn every N experiences
            batch_size: Batch size for learning
            update_frequency: Frequency of model updates
        """
        self.learn_rate = learn_rate
        self.batch_size = batch_size
        self.update_frequency = update_frequency
        self.experience_counter = 0
    
    def should_learn(self) -> bool:
        """Check if learning should occur.
        
        Returns:
            True if learning should happen
        """
        self.experience_counter += 1
        return self.experience_counter % self.learn_rate == 0
    
    def reset(self):
        """Reset scheduler."""
        self.experience_counter = 0
