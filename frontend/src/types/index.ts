// Core domain types
export interface Node {
  id: string;
  label: string;
  x?: number;
  y?: number;
}

export interface Link {
  id: string;
  source: string;
  target: string;
  capacity: number;
  delay: number;
  utilization?: number;
  weight?: number;
}

export interface Topology {
  id: string;
  name: string;
  nodes: Node[];
  links: Link[];
  createdAt: string;
  updatedAt: string;
}

export interface TrafficFlow {
  id: string;
  source: string;
  destination: string;
  bandwidth: number;
  tos: number;
  avgBw: number;
  pktsGen: number;
  timeDist: string;
  sizeDist: string;
}

// Experiment types
export type ExperimentType = 'inference' | 'training';
export type ExperimentStatus = 'idle' | 'running' | 'paused' | 'completed' | 'error' | 'stopped' | 'failed';

export interface InferenceConfig {
  topologyId: string;
  flows: TrafficFlow[];
  duration: number;
  modelVersion?: string;
  routingAlgorithm?: string;
}

export interface TrainingConfig {
  topologyId: string;
  episodes: number;
  batchSize: number;
  learningRate: number;
  gamma: number;
  epsilonStart: number;
  epsilonEnd: number;
  epsilonDecay: number;
  checkpointInterval: number;
}

export interface Experiment {
  id: string;
  type: ExperimentType;
  status: ExperimentStatus;
  config: InferenceConfig | TrainingConfig;
  startTime?: string;
  endTime?: string;
  progress: number;
  currentEpisode?: number;
  totalEpisodes?: number;
  metrics?: MetricsSummary;
}

// Metrics types
export interface QoSMetrics {
  avgDelay: number;
  maxDelay: number;
  jitter: number;
  packetLoss: number;
  throughput: number;
}

export interface UtilizationMetrics {
  linkId: string;
  utilization: number;
  congestion: boolean;
}

export interface LearningMetrics {
  episode: number;
  reward: number;
  loss: number;
  epsilon: number;
  steps: number;
}

export interface MetricsSummary {
  qos: QoSMetrics;
  utilization: UtilizationMetrics[];
  fairness: number;
  slaViolations: number;
}

export interface MetricsTimeSeries {
  timestamp: number;
  delay: number;
  jitter: number;
  packetLoss: number;
  fairness: number;
  reward?: number;
}

// Model types
export interface ModelVersion {
  id: string;
  version: string;
  name: string;
  active: boolean;
  createdAt: string;
  checkpointPath: string;
  metrics?: {
    avgReward: number;
    episodes: number;
    accuracy?: number;
  };
  metadata?: Record<string, any>;
}

export interface Checkpoint {
  episode: number;
  timestamp: string;
  path: string;
  metrics: {
    reward: number;
    loss: number;
  };
}

// System health types
export interface SystemHealth {
  status: 'healthy' | 'degraded' | 'down';
  uptime: number;
  version: string;
  lastCheck: string;
  components: {
    api: boolean;
    database: boolean;
    simulator: boolean;
    learningModule: boolean;
  };
}

export interface APIEndpoint {
  method: string;
  path: string;
  description: string;
  parameters?: Record<string, any>;
}

export interface APIStats {
  endpoint: string;
  avgLatency: number;
  requestCount: number;
  errorRate: number;
}

// WebSocket message types
export interface WSMessage {
  type: 'progress' | 'metrics' | 'status' | 'error' | 'checkpoint';
  experimentId?: string;
  data: any;
  timestamp: string;
}

// API response types
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
}

// UI state types
export interface FilterOptions {
  status?: ExperimentStatus[];
  type?: ExperimentType[];
  dateRange?: [string, string];
  search?: string;
}

export interface SortOptions {
  field: string;
  order: 'asc' | 'desc';
}

export interface ChartConfig {
  id: string;
  type: 'line' | 'bar' | 'scatter' | 'heatmap' | 'gauge';
  title: string;
  xAxis?: string;
  yAxis?: string;
  series: string[];
}

export interface ExportOptions {
  format: 'csv' | 'json' | 'png';
  includeMetrics: boolean;
  includeTopology: boolean;
  includeConfig: boolean;
}
