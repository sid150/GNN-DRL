import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { api } from '@/services/api';
import type { SystemHealth } from '@/types';

interface SystemState {
  health: SystemHealth | null;
  version: string;
  theme: 'light' | 'dark';
  wsConnected: boolean;
  notifications: Notification[];
  loading: boolean;
  error: string | null;
}

interface Notification {
  id: string;
  type: 'info' | 'success' | 'warning' | 'error';
  message: string;
  timestamp: string;
}

const initialState: SystemState = {
  health: null,
  version: '',
  theme: (localStorage.getItem('theme') as 'light' | 'dark') || 'light',
  wsConnected: false,
  notifications: [],
  loading: false,
  error: null,
};

// Async thunks
export const fetchSystemHealth = createAsyncThunk('system/fetchHealth', async () => {
  return await api.health.getStatus();
});

export const fetchVersion = createAsyncThunk('system/fetchVersion', async () => {
  return await api.health.getVersion();
});

const systemSlice = createSlice({
  name: 'system',
  initialState,
  reducers: {
    setTheme: (state, action: PayloadAction<'light' | 'dark'>) => {
      state.theme = action.payload;
      localStorage.setItem('theme', action.payload);
    },
    setWsConnected: (state, action: PayloadAction<boolean>) => {
      state.wsConnected = action.payload;
    },
    addNotification: (state, action: PayloadAction<Omit<Notification, 'id' | 'timestamp'>>) => {
      state.notifications.push({
        id: Date.now().toString(),
        timestamp: new Date().toISOString(),
        ...action.payload,
      });
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter((n) => n.id !== action.payload);
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch health
      .addCase(fetchSystemHealth.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(fetchSystemHealth.fulfilled, (state, action) => {
        state.loading = false;
        state.health = action.payload;
      })
      .addCase(fetchSystemHealth.rejected, (state, action) => {
        state.loading = false;
        state.error = action.error.message || 'Failed to fetch system health';
      })
      // Fetch version
      .addCase(fetchVersion.fulfilled, (state, action) => {
        if (action.payload) {
          state.version = action.payload.version;
        }
      })
      .addCase(fetchVersion.rejected, (state, _action) => {
        // Version fetch failed, but don't block the app
        state.version = 'Unknown';
      });
  },
});

export const {
  setTheme,
  setWsConnected,
  addNotification,
  removeNotification,
  clearNotifications,
  clearError,
} = systemSlice.actions;

export default systemSlice.reducer;
