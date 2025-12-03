import { useEffect } from 'react';
import { Grid, Paper, Typography, Box, Chip } from '@mui/material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchSystemHealth } from '../store/slices/systemSlice';

export default function Health() {
  const dispatch = useAppDispatch();
  const { health, version } = useAppSelector((state) => state.system);

  useEffect(() => {
    dispatch(fetchSystemHealth());
  }, [dispatch]);

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        System Health & API
      </Typography>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              System Status
            </Typography>
            <Box sx={{ mt: 2 }}>
              <Typography variant="body1">
                Status:{' '}
                <Chip
                  label={health?.status || 'Unknown'}
                  color={health?.status === 'healthy' ? 'success' : 'error'}
                  size="small"
                />
              </Typography>
              <Typography variant="body1" sx={{ mt: 1 }}>
                Version: {version || 'Unknown'}
              </Typography>
              <Typography variant="body1" sx={{ mt: 1 }}>
                Uptime: {health?.uptime ? `${Math.floor(health.uptime / 3600)}h` : 'N/A'}
              </Typography>
            </Box>
          </Paper>
        </Grid>

        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Components
            </Typography>
            <Box sx={{ mt: 2 }}>
              {health?.components && Object.entries(health.components).map(([key, value]) => (
                <Box key={key} sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                  <Typography>{key}</Typography>
                  <Chip
                    label={value ? 'Online' : 'Offline'}
                    color={value ? 'success' : 'error'}
                    size="small"
                  />
                </Box>
              ))}
            </Box>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
