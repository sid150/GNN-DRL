import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { api } from '@/services/api';
import type { Experiment, ExperimentStatus, FilterOptions, SortOptions } from '@/types';

interface ExperimentsState {
  items: Experiment[];
  activeExperiment: Experiment | null;
  loading: boolean;
  error: string | null;
  filters: FilterOptions;
  sort: SortOptions;
}

const initialState: ExperimentsState = {
  items: [],
  activeExperiment: null,
  loading: false,
  error: null,
  filters: {},
  sort: { field: 'startTime', order: 'desc' },
};

// Async thunks
export const startInference = createAsyncThunk(
  'experiments/startInference',
  async (config: any) => {
    return await api.inference.start(config);
  }
);

export const startTraining = createAsyncThunk(
  'experiments/startTraining',
  async (config: any) => {
    return await api.training.start(config);
  }
);

export const stopExperiment = createAsyncThunk(
  'experiments/stop',
  async ({ id, type }: { id: string; type: 'inference' | 'training' }) => {
    if (type === 'inference') {
      await api.inference.stop(id);
    } else {
      await api.training.stop(id);
    }
    return id;
  }
);

export const fetchExperimentStatus = createAsyncThunk(
  'experiments/fetchStatus',
  async ({ id, type }: { id: string; type: 'inference' | 'training' }) => {
    if (type === 'inference') {
      return await api.inference.getStatus(id);
    } else {
      return await api.training.getStatus(id);
    }
  }
);

const experimentsSlice = createSlice({
  name: 'experiments',
  initialState,
  reducers: {
    setActiveExperiment: (state, action: PayloadAction<Experiment | null>) => {
      state.activeExperiment = action.payload;
    },
    updateExperimentStatus: (state, action: PayloadAction<{ id: string; status: ExperimentStatus }>) => {
      const experiment = state.items.find((e) => e.id === action.payload.id);
      if (experiment) {
        experiment.status = action.payload.status;
      }
      if (state.activeExperiment?.id === action.payload.id) {
        state.activeExperiment.status = action.payload.status;
      }
    },
    updateExperimentProgress: (state, action: PayloadAction<{ id: string; progress: number }>) => {
      const experiment = state.items.find((e) => e.id === action.payload.id);
      if (experiment) {
        experiment.progress = action.payload.progress;
      }
      if (state.activeExperiment?.id === action.payload.id) {
        state.activeExperiment.progress = action.payload.progress;
      }
    },
    setFilters: (state, action: PayloadAction<FilterOptions>) => {
      state.filters = action.payload;
    },
    setSort: (state, action: PayloadAction<SortOptions>) => {
      state.sort = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Start inference
      .addCase(startInference.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(startInference.fulfilled, (state, action) => {
        state.loading = false;
        state.items.unshift(action.payload);
        state.activeExperiment = action.payload;
      })
      .addCase(startInference.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to start inference';
      })
      // Start training
      .addCase(startTraining.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(startTraining.fulfilled, (state, action) => {
        state.loading = false;
        state.items.unshift(action.payload);
        state.activeExperiment = action.payload;
      })
      .addCase(startTraining.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to start training';
      })
      // Stop experiment
      .addCase(stopExperiment.fulfilled, (state, action) => {
        const experiment = state.items.find((e) => e.id === action.payload);
        if (experiment) {
          experiment.status = 'completed';
        }
      })
      // Fetch status
      .addCase(fetchExperimentStatus.fulfilled, (state, action) => {
        const index = state.items.findIndex((e) => e.id === action.payload.id);
        if (index !== -1) {
          state.items[index] = action.payload;
        }
        if (state.activeExperiment?.id === action.payload.id) {
          state.activeExperiment = action.payload;
        }
      });
  },
});

export const {
  setActiveExperiment,
  updateExperimentStatus,
  updateExperimentProgress,
  setFilters,
  setSort,
  clearError,
} = experimentsSlice.actions;

export default experimentsSlice.reducer;
