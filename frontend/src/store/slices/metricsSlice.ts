import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { api } from '@/services/api';
import type { MetricsSummary, MetricsTimeSeries } from '@/types';

interface MetricsState {
  summaries: Record<string, MetricsSummary>;
  timeSeries: Record<string, MetricsTimeSeries[]>;
  selectedExperimentIds: string[];
  loading: boolean;
  error: string | null;
}

const initialState: MetricsState = {
  summaries: {},
  timeSeries: {},
  selectedExperimentIds: [],
  loading: false,
  error: null,
};

// Async thunks
export const fetchMetrics = createAsyncThunk('metrics/fetch', async (experimentId: string) => {
  const [summary, series] = await Promise.all([
    api.inference.getResults(experimentId).catch(() => null),
    api.metrics.get(experimentId).catch(() => []),
  ]);
  return { experimentId, summary, series };
});

export const fetchMultipleMetrics = createAsyncThunk('metrics/fetchMultiple', async (experimentIds: string[]) => {
  return await api.metrics.aggregate(experimentIds);
});

const metricsSlice = createSlice({
  name: 'metrics',
  initialState,
  reducers: {
    selectExperiments: (state, action: PayloadAction<string[]>) => {
      state.selectedExperimentIds = action.payload;
    },
    clearMetrics: (state) => {
      state.summaries = {};
      state.timeSeries = {};
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      .addCase(fetchMetrics.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchMetrics.fulfilled, (state, action) => {
        state.loading = false;
        const { experimentId, summary, series } = action.payload;
        if (summary) {
          state.summaries[experimentId] = summary;
        }
        state.timeSeries[experimentId] = series;
      })
      .addCase(fetchMetrics.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch metrics';
      })
      .addCase(fetchMultipleMetrics.fulfilled, (state, action) => {
        state.summaries = { ...state.summaries, ...action.payload };
      });
  },
});

export const { selectExperiments, clearMetrics, clearError } = metricsSlice.actions;

export default metricsSlice.reducer;
