import { useEffect, useState } from 'react';
import { Grid, Paper, Typography, Box, Card, CardContent, Button, Dialog, DialogTitle, DialogContent, DialogActions, TextField, MenuItem, Alert } from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  CheckCircle as CheckIcon,
} from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '../store/hooks';
import { fetchTopologies } from '../store/slices/topologiesSlice';
import { fetchModels } from '../store/slices/modelsSlice';
import { apiClient } from '../services/apiClient';

export default function Dashboard() {
  const dispatch = useAppDispatch();
  const { health } = useAppSelector((state) => state.system);
  const { items: experiments, activeExperiment } = useAppSelector((state) => state.experiments);
  const { activeModel } = useAppSelector((state) => state.models);
  
  const [openBuildDialog, setOpenBuildDialog] = useState(false);
  const [buildConfig, setBuildConfig] = useState({
    topology_type: 'nsfnet',
    num_nodes: 14,
  });
  const [buildLoading, setBuildLoading] = useState(false);
  const [buildSuccess, setBuildSuccess] = useState(false);
  const [buildError, setBuildError] = useState<string | null>(null);

  useEffect(() => {
    dispatch(fetchTopologies());
    dispatch(fetchModels());
  }, [dispatch]);

  const handleBuildNetwork = async () => {
    setBuildLoading(true);
    setBuildError(null);
    try {
      await apiClient.post('/api/v1/topology/create', buildConfig);
      setBuildSuccess(true);
      setTimeout(() => {
        setBuildSuccess(false);
        setOpenBuildDialog(false);
        dispatch(fetchTopologies());
      }, 1500);
    } catch (error: any) {
      setBuildError(error.message || 'Failed to build network');
    } finally {
      setBuildLoading(false);
    }
  };

  const topologyTypes = [
    { value: 'nsfnet', label: 'NSFNET (14 nodes)', defaultNodes: 14, fixedNodes: true },
    { value: 'geant2', label: 'GEANT2 (24 nodes)', defaultNodes: 24, fixedNodes: true },
    { value: 'random', label: 'Random Topology', defaultNodes: 10, fixedNodes: false },
  ];

  const selectedTopologyType = topologyTypes.find(t => t.value === buildConfig.topology_type);

  const recentExperiments = experiments.slice(0, 5);

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        Dashboard
      </Typography>

      <Grid container spacing={3}>
        {/* System Status Cards */}
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                System Status
              </Typography>
              <Typography variant="h5" component="div">
                {health?.status === 'healthy' ? (
                  <Box sx={{ color: 'success.main', display: 'flex', alignItems: 'center' }}>
                    <CheckIcon sx={{ mr: 1 }} /> Healthy
                  </Box>
                ) : (
                  'Unknown'
                )}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Active Experiments
              </Typography>
              <Typography variant="h5">
                {experiments.filter((e) => e.status === 'running').length}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Total Experiments
              </Typography>
              <Typography variant="h5">{experiments.length}</Typography>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography color="textSecondary" gutterBottom>
                Active Model
              </Typography>
              <Typography variant="h5" sx={{ fontSize: '1rem' }}>
                {activeModel?.version || 'None'}
              </Typography>
            </CardContent>
          </Card>
        </Grid>

        {/* Current Experiment */}
        {activeExperiment && (
          <Grid item xs={12}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Current Experiment
              </Typography>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Typography>ID: {activeExperiment.id}</Typography>
                <Typography>Type: {activeExperiment.type}</Typography>
                <Typography>Status: {activeExperiment.status}</Typography>
                <Typography>Progress: {activeExperiment.progress}%</Typography>
              </Box>
            </Paper>
          </Grid>
        )}

        {/* Quick Actions */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Quick Actions
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
              <Button
                variant="contained"
                startIcon={<PlayIcon />}
                onClick={() => window.location.href = '/experiments'}
              >
                Start New Inference
              </Button>
              <Button
                variant="contained"
                color="secondary"
                startIcon={<PlayIcon />}
                onClick={() => window.location.href = '/experiments'}
              >
                Start Training
              </Button>
              <Button
                variant="outlined"
                onClick={() => setOpenBuildDialog(true)}
              >
                Build Network
              </Button>
            </Box>
          </Paper>
        </Grid>

        {/* Recent Experiments */}
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Recent Experiments
            </Typography>
            <Box sx={{ mt: 2 }}>
              {recentExperiments.length === 0 ? (
                <Typography color="textSecondary">No experiments yet</Typography>
              ) : (
                recentExperiments.map((exp) => (
                  <Box
                    key={exp.id}
                    sx={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      alignItems: 'center',
                      py: 1,
                      borderBottom: '1px solid',
                      borderColor: 'divider',
                    }}
                  >
                    <Typography variant="body2">{exp.id.slice(0, 8)}...</Typography>
                    <Typography variant="body2" color="textSecondary">
                      {exp.type}
                    </Typography>
                    <Typography
                      variant="body2"
                      sx={{
                        color:
                          exp.status === 'completed'
                            ? 'success.main'
                            : exp.status === 'error'
                            ? 'error.main'
                            : 'primary.main',
                      }}
                    >
                      {exp.status}
                    </Typography>
                  </Box>
                ))
              )}
            </Box>
          </Paper>
        </Grid>
      </Grid>

      {/* Build Network Dialog */}
      <Dialog open={openBuildDialog} onClose={() => setOpenBuildDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Build Network Topology</DialogTitle>
        <DialogContent>
          {buildSuccess && (
            <Alert severity="success" sx={{ mb: 2 }}>
              Network topology created successfully!
            </Alert>
          )}
          {buildError && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {buildError}
            </Alert>
          )}
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <TextField
              select
              label="Topology Type"
              value={buildConfig.topology_type}
              onChange={(e) => {
                const selectedType = topologyTypes.find(t => t.value === e.target.value);
                setBuildConfig({
                  topology_type: e.target.value,
                  num_nodes: selectedType?.defaultNodes || 10,
                });
              }}
              fullWidth
            >
              {topologyTypes.map((type) => (
                <MenuItem key={type.value} value={type.value}>
                  {type.label}
                </MenuItem>
              ))}
            </TextField>

            <TextField
              label="Number of Nodes"
              type="number"
              value={buildConfig.num_nodes}
              onChange={(e) => setBuildConfig({ ...buildConfig, num_nodes: parseInt(e.target.value) })}
              fullWidth
              disabled={selectedTopologyType?.fixedNodes}
              helperText={
                selectedTopologyType?.fixedNodes 
                  ? "Node count is fixed for this topology type"
                  : "Adjust the number of nodes for the topology"
              }
              inputProps={{ min: 2, max: 100 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenBuildDialog(false)} disabled={buildLoading}>
            Cancel
          </Button>
          <Button onClick={handleBuildNetwork} variant="contained" disabled={buildLoading}>
            {buildLoading ? 'Building...' : 'Build Network'}
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}
