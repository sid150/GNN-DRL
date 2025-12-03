# GNN-DRL: Graph Neural Network with Deep Reinforcement Learning for Network Routing

A full-stack application that uses Graph Neural Networks (GNN) with Deep Reinforcement Learning (DRL) to optimize network routing decisions.

## Quick Start

### Backend Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the backend API server:**
   ```bash
   cd backend
   python main.py api --port 8000
   ```
   
   The backend will be available at `http://localhost:8000`

### Frontend Setup

1. **Setup and start the frontend server:**
   ```bash
   cd frontend
   ./setup.sh
   ./start.sh
   ```
   
   The frontend will be available at `http://localhost:3000`

## Usage

1. Open your browser to `http://localhost:3000`
2. Navigate to the **Experiments** tab
3. **Training**: Configure and start a training experiment to train the GNN model
4. **Inference**: Run inference with trained models to see routing performance metrics
5. View comprehensive metrics including QoS, network utilization, learning metrics, and operational overhead

