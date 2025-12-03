import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { api } from '@/services/api';
import type { ModelVersion } from '@/types';

interface ModelsState {
  items: ModelVersion[];
  activeModel: ModelVersion | null;
  loading: boolean;
  error: string | null;
}

const initialState: ModelsState = {
  items: [],
  activeModel: null,
  loading: false,
  error: null,
};

// Async thunks
export const fetchModels = createAsyncThunk('models/fetchAll', async () => {
  return await api.models.list();
});

export const loadModel = createAsyncThunk('models/load', async (checkpointPath: string) => {
  return await api.models.load(checkpointPath);
});

export const saveModel = createAsyncThunk('models/save', async ({ name, metadata }: { name: string; metadata?: Record<string, any> }) => {
  return await api.models.save(name, metadata);
});

export const deleteModel = createAsyncThunk('models/delete', async (id: string) => {
  await api.models.delete(id);
  return id;
});

export const activateModel = createAsyncThunk('models/activate', async (id: string) => {
  await api.models.activate(id);
  return id;
});

export const deactivateModel = createAsyncThunk('models/deactivate', async (id: string) => {
  await api.models.deactivate(id);
  return id;
});

const modelsSlice = createSlice({
  name: 'models',
  initialState,
  reducers: {
    setActiveModel: (state, action: PayloadAction<ModelVersion | null>) => {
      state.activeModel = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch all
      .addCase(fetchModels.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchModels.fulfilled, (state, action) => {
        state.loading = false;
        state.items = action.payload;
        state.activeModel = action.payload.find((m) => m.active) || null;
      })
      .addCase(fetchModels.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch models';
      })
      // Load
      .addCase(loadModel.fulfilled, (state, action) => {
        state.items.push(action.payload);
      })
      // Save
      .addCase(saveModel.fulfilled, (state, action) => {
        state.items.push(action.payload);
      })
      // Delete
      .addCase(deleteModel.fulfilled, (state, action) => {
        state.items = state.items.filter((m) => m.id !== action.payload);
        if (state.activeModel?.id === action.payload) {
          state.activeModel = null;
        }
      })
      // Activate
      .addCase(activateModel.fulfilled, (state, action) => {
        state.items.forEach((m) => {
          m.active = m.id === action.payload;
        });
        const activated = state.items.find((m) => m.id === action.payload);
        if (activated) {
          state.activeModel = activated;
        }
      })
      // Deactivate
      .addCase(deactivateModel.fulfilled, (state, action) => {
        const model = state.items.find((m) => m.id === action.payload);
        if (model) {
          model.active = false;
        }
        if (state.activeModel?.id === action.payload) {
          state.activeModel = null;
        }
      });
  },
});

export const { setActiveModel, clearError } = modelsSlice.actions;

export default modelsSlice.reducer;
