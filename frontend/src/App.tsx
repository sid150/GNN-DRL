import { useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Box } from '@mui/material';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import Experiments from './pages/Experiments';
import Topologies from './pages/Topologies';
import Metrics from './pages/Metrics';
import Models from './pages/Models';
import Health from './pages/Health';
import Settings from './pages/Settings';
import { useAppDispatch } from './store/hooks';
import { fetchSystemHealth, fetchVersion } from './store/slices/systemSlice';

function App() {
  const dispatch = useAppDispatch();

  useEffect(() => {
    // Fetch initial system data
    dispatch(fetchSystemHealth());
    dispatch(fetchVersion());

    // Set up polling for system health
    const interval = setInterval(() => {
      dispatch(fetchSystemHealth());
    }, 30000); // Every 30 seconds

    return () => clearInterval(interval);
  }, [dispatch]);

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <Layout>
        <Routes>
          <Route path="/" element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/topologies" element={<Topologies />} />
          <Route path="/metrics" element={<Metrics />} />
          <Route path="/models" element={<Models />} />
          <Route path="/health" element={<Health />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </Layout>
    </Box>
  );
}

export default App;
