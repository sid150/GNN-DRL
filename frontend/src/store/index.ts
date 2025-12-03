import { configureStore } from '@reduxjs/toolkit';
import experimentsReducer from './slices/experimentsSlice';
import topologiesReducer from './slices/topologiesSlice';
import metricsReducer from './slices/metricsSlice';
import modelsReducer from './slices/modelsSlice';
import systemReducer from './slices/systemSlice';

export const store = configureStore({
  reducer: {
    experiments: experimentsReducer,
    topologies: topologiesReducer,
    metrics: metricsReducer,
    models: modelsReducer,
    system: systemReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: {
        // Ignore these action types
        ignoredActions: ['experiments/updateProgress'],
      },
    }),
});

export type RootState = ReturnType<typeof store.getState>;
export type AppDispatch = typeof store.dispatch;
