import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { api } from '@/services/api';
import type { Topology } from '@/types';

interface TopologiesState {
  items: Topology[];
  selectedTopology: Topology | null;
  loading: boolean;
  error: string | null;
}

const initialState: TopologiesState = {
  items: [],
  selectedTopology: null,
  loading: false,
  error: null,
};

// Async thunks
export const fetchTopologies = createAsyncThunk('topologies/fetchAll', async () => {
  return await api.topology.list();
});

export const fetchTopology = createAsyncThunk('topologies/fetchOne', async (id: string) => {
  return await api.topology.get(id);
});

export const createTopology = createAsyncThunk('topologies/create', async (topology: any) => {
  return await api.topology.create(topology);
});

export const importTopology = createAsyncThunk('topologies/import', async (file: File) => {
  return await api.topology.import(file);
});

export const deleteTopology = createAsyncThunk('topologies/delete', async (id: string) => {
  await api.topology.delete(id);
  return id;
});

const topologiesSlice = createSlice({
  name: 'topologies',
  initialState,
  reducers: {
    setSelectedTopology: (state, action: PayloadAction<Topology | null>) => {
      state.selectedTopology = action.payload;
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch all
      .addCase(fetchTopologies.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchTopologies.fulfilled, (state, action) => {
        state.loading = false;
        state.items = action.payload;
      })
      .addCase(fetchTopologies.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch topologies';
      })
      // Fetch one
      .addCase(fetchTopology.fulfilled, (state, action) => {
        state.selectedTopology = action.payload;
      })
      // Create
      .addCase(createTopology.fulfilled, (state, action) => {
        state.items.push(action.payload);
      })
      // Import
      .addCase(importTopology.fulfilled, (state, action) => {
        state.items.push(action.payload);
      })
      // Delete
      .addCase(deleteTopology.fulfilled, (state, action) => {
        state.items = state.items.filter((t) => t.id !== action.payload);
        if (state.selectedTopology?.id === action.payload) {
          state.selectedTopology = null;
        }
      });
  },
});

export const { setSelectedTopology, clearError } = topologiesSlice.actions;

export default topologiesSlice.reducer;
