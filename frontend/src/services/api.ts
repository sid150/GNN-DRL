import { apiClient } from './apiClient';
import { config } from '@/config';
import type {
  Topology,
  Experiment,
  InferenceConfig,
  TrainingConfig,
  ModelVersion,
  Checkpoint,
  SystemHealth,
  MetricsSummary,
  MetricsTimeSeries,
  PaginatedResponse,
  ExportOptions,
} from '@/types';

// Health API
export const healthAPI = {
  async getStatus(): Promise<SystemHealth> {
    const response = await apiClient.get<SystemHealth>(config.endpoints.health);
    return response.data!;
  },

  async getVersion(): Promise<{ version: string; buildDate: string }> {
    const response = await apiClient.get<{ version: string; buildDate: string }>(config.endpoints.version);
    return response.data!;
  },
};

// Inference API
export const inferenceAPI = {
  async start(inferenceConfig: InferenceConfig): Promise<Experiment> {
    const response = await apiClient.post<Experiment>(config.endpoints.inferenceStart, inferenceConfig);
    return response.data!;
  },

  async stop(experimentId: string): Promise<void> {
    await apiClient.post(`${config.endpoints.inferenceStop}/${experimentId}`);
  },

  async getStatus(experimentId: string): Promise<Experiment> {
    const response = await apiClient.get<Experiment>(`${config.endpoints.inferenceStatus}/${experimentId}`);
    return response.data!;
  },

  async getResults(experimentId: string): Promise<MetricsSummary> {
    const response = await apiClient.get<MetricsSummary>(`${config.endpoints.inferenceResults}/${experimentId}`);
    return response.data!;
  },
};

// Training API
export const trainingAPI = {
  async start(trainingConfig: TrainingConfig): Promise<Experiment> {
    const response = await apiClient.post<Experiment>(config.endpoints.trainStart, trainingConfig);
    return response.data!;
  },

  async stop(experimentId: string): Promise<void> {
    await apiClient.post(`${config.endpoints.trainStop}/${experimentId}`);
  },

  async getStatus(experimentId: string): Promise<Experiment> {
    const response = await apiClient.get<Experiment>(`${config.endpoints.trainStatus}/${experimentId}`);
    return response.data!;
  },

  async getCheckpoints(experimentId: string): Promise<Checkpoint[]> {
    const response = await apiClient.get<Checkpoint[]>(`${config.endpoints.trainCheckpoints}/${experimentId}`);
    return response.data!;
  },

  async resume(experimentId: string, checkpointPath: string): Promise<Experiment> {
    const response = await apiClient.post<Experiment>(config.endpoints.trainResume, {
      experimentId,
      checkpointPath,
    });
    return response.data!;
  },
};

// Topology API
export const topologyAPI = {
  async list(): Promise<Topology[]> {
    const response = await apiClient.get<Topology[]>(config.endpoints.topologyList);
    return response.data!;
  },

  async get(id: string): Promise<Topology> {
    const response = await apiClient.get<Topology>(`${config.endpoints.topologyGet}/${id}`);
    return response.data!;
  },

  async create(topology: Omit<Topology, 'id' | 'createdAt' | 'updatedAt'>): Promise<Topology> {
    const response = await apiClient.post<Topology>(config.endpoints.topologyCreate, topology);
    return response.data!;
  },

  async import(file: File): Promise<Topology> {
    const response = await apiClient.upload(config.endpoints.topologyImport, file);
    return response.data!;
  },

  async export(id: string, format: 'json' | 'graphml' | 'gml'): Promise<void> {
    await apiClient.download(`${config.endpoints.topologyExport}/${id}?format=${format}`, `topology-${id}.${format}`);
  },

  async delete(id: string): Promise<void> {
    await apiClient.delete(`${config.endpoints.topologyDelete}/${id}`);
  },
};

// Metrics API
export const metricsAPI = {
  async list(experimentId?: string): Promise<PaginatedResponse<MetricsSummary>> {
    const url = experimentId 
      ? `${config.endpoints.metricsList}?experimentId=${experimentId}`
      : config.endpoints.metricsList;
    const response = await apiClient.get<PaginatedResponse<MetricsSummary>>(url);
    return response.data!;
  },

  async get(experimentId: string): Promise<MetricsTimeSeries[]> {
    const response = await apiClient.get<MetricsTimeSeries[]>(`${config.endpoints.metricsGet}/${experimentId}`);
    return response.data!;
  },

  async aggregate(experimentIds: string[]): Promise<{ [key: string]: MetricsSummary }> {
    const response = await apiClient.post<{ [key: string]: MetricsSummary }>(
      config.endpoints.metricsAggregate,
      { experimentIds }
    );
    return response.data!;
  },

  async correlation(experimentId: string): Promise<{ [key: string]: number }> {
    const response = await apiClient.get<{ [key: string]: number }>(
      `${config.endpoints.metricsCorrelation}/${experimentId}`
    );
    return response.data!;
  },

  async export(experimentId: string, options: ExportOptions): Promise<void> {
    const format = options.format;
    await apiClient.download(
      `${config.endpoints.metricsExport}/${experimentId}?format=${format}`,
      `metrics-${experimentId}.${format}`
    );
  },
};

// Models API
export const modelsAPI = {
  async list(): Promise<ModelVersion[]> {
    const response = await apiClient.get<ModelVersion[]>(config.endpoints.modelsList);
    return response.data!;
  },

  async load(checkpointPath: string): Promise<ModelVersion> {
    const response = await apiClient.post<ModelVersion>(config.endpoints.modelsLoad, { checkpointPath });
    return response.data!;
  },

  async save(name: string, metadata?: Record<string, any>): Promise<ModelVersion> {
    const response = await apiClient.post<ModelVersion>(config.endpoints.modelsSave, { name, metadata });
    return response.data!;
  },

  async delete(id: string): Promise<void> {
    await apiClient.delete(`${config.endpoints.modelsDelete}/${id}`);
  },

  async activate(id: string): Promise<void> {
    await apiClient.post(`${config.endpoints.modelsActivate}/${id}`);
  },

  async deactivate(id: string): Promise<void> {
    await apiClient.post(`${config.endpoints.modelsDeactivate}/${id}`);
  },
};

// Export all APIs
export const api = {
  health: healthAPI,
  inference: inferenceAPI,
  training: trainingAPI,
  topology: topologyAPI,
  metrics: metricsAPI,
  models: modelsAPI,
};
