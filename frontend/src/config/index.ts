export const config = {
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  wsBaseUrl: import.meta.env.VITE_WS_BASE_URL || 'ws://localhost:8000',
  appTitle: import.meta.env.VITE_APP_TITLE || 'GNN-DRL Network Optimizer',
  
  // Feature flags
  enableWebSocket: import.meta.env.VITE_ENABLE_WEBSOCKET === 'true',
  enableAuth: import.meta.env.VITE_ENABLE_AUTH === 'true',
  enableExperiments: import.meta.env.VITE_ENABLE_EXPERIMENTS !== 'false',
  enableTraining: import.meta.env.VITE_ENABLE_TRAINING !== 'false',
  
  // Performance settings
  pollingInterval: parseInt(import.meta.env.VITE_POLLING_INTERVAL) || 5000,
  maxTopologyNodes: parseInt(import.meta.env.VITE_MAX_TOPOLOGY_NODES) || 1000,
  chartMaxPoints: parseInt(import.meta.env.VITE_CHART_MAX_POINTS) || 1000,
  tablePageSize: parseInt(import.meta.env.VITE_TABLE_PAGE_SIZE) || 25,
  
  // API endpoints
  endpoints: {
    // Health
    health: '/health',
    version: '/version',
    
    // Inference
    inferenceStart: '/inference/start',
    inferenceStop: '/inference/stop',
    inferenceStatus: '/inference/status',
    inferenceResults: '/inference/results',
    
    // Training
    trainStart: '/train/start',
    trainStop: '/train/stop',
    trainStatus: '/train/status',
    trainCheckpoints: '/train/checkpoints',
    trainResume: '/train/resume',
    
    // Topology
    topologyList: '/topology/list',
    topologyGet: '/topology/get',
    topologyCreate: '/topology/create',
    topologyImport: '/topology/import',
    topologyExport: '/topology/export',
    topologyDelete: '/topology/delete',
    
    // Metrics
    metricsList: '/metrics/list',
    metricsGet: '/metrics/get',
    metricsAggregate: '/metrics/aggregate',
    metricsCorrelation: '/metrics/correlation',
    metricsExport: '/metrics/export',
    
    // Models
    modelsList: '/models/list',
    modelsLoad: '/models/load',
    modelsSave: '/models/save',
    modelsDelete: '/models/delete',
    modelsActivate: '/models/activate',
    modelsDeactivate: '/models/deactivate',
  },
} as const;

export type Config = typeof config;
