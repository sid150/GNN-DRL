import { useState, useEffect } from 'react';
import {
  Typography,
  Box,
  Card,
  Grid,
  Button,
  TextField,
  MenuItem,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  LinearProgress,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Alert,
  Collapse,
  IconButton,
  Divider,
} from '@mui/material';
import { Stop, Science, TrendingUp, KeyboardArrowDown, KeyboardArrowUp } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import { startTraining, startInference, stopExperiment, fetchExperimentStatus } from '@/store/slices/experimentsSlice';
import type { Experiment } from '@/types';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;
  return (
    <div role="tabpanel" hidden={value !== index} {...other}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

export default function Experiments() {
  const dispatch = useAppDispatch();
  const { items: experiments, loading, error } = useAppSelector((state) => state.experiments);
  const topologies = useAppSelector((state: any) => state.topologies.items);
  const models = useAppSelector((state: any) => state.models.items);

  const [tabValue, setTabValue] = useState(0);
  const [openTrainingDialog, setOpenTrainingDialog] = useState(false);
  const [openInferenceDialog, setOpenInferenceDialog] = useState(false);

  // Training form state
  const [trainingConfig, setTrainingConfig] = useState({
    topologyId: 'nsfnet',
    episodes: 100,
    saveInterval: 10,
    learningRate: 0.001,
    batchSize: 32,
    gamma: 0.99,
  });

  // Inference form state
  const [inferenceConfig, setInferenceConfig] = useState({
    topologyId: 'nsfnet',
    modelVersion: 'best',
    numFlows: 20,
    duration: 100,
  });

  // Poll for experiment status updates
  useEffect(() => {
    const runningExperiments = experiments.filter((exp) => exp.status === 'running');
    
    if (runningExperiments.length > 0) {
      const interval = setInterval(() => {
        runningExperiments.forEach((exp) => {
          dispatch(
            exp.type === 'training'
              ? fetchExperimentStatus({ id: exp.id, type: 'training' })
              : fetchExperimentStatus({ id: exp.id, type: 'inference' })
          );
        });
      }, 2000); // Poll every 2 seconds

      return () => clearInterval(interval);
    }
  }, [experiments, dispatch]);

  const handleTabChange = (_event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const handleStartTraining = async () => {
    await dispatch(startTraining(trainingConfig));
    setOpenTrainingDialog(false);
  };

  const handleStartInference = async () => {
    await dispatch(startInference(inferenceConfig));
    setOpenInferenceDialog(false);
  };

  const handleStopExperiment = (experiment: Experiment) => {
    dispatch(stopExperiment({ id: experiment.id, type: experiment.type }));
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'stopped':
        return 'warning';
      default:
        return 'default';
    }
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Experiments</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button
            variant="contained"
            startIcon={<TrendingUp />}
            onClick={() => setOpenTrainingDialog(true)}
            color="primary"
          >
            Start Training
          </Button>
          <Button
            variant="outlined"
            startIcon={<Science />}
            onClick={() => setOpenInferenceDialog(true)}
          >
            Start Inference
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }}>
          {error}
        </Alert>
      )}

      <Card>
        <Tabs value={tabValue} onChange={handleTabChange}>
          <Tab label="All Experiments" />
          <Tab label="Training" />
          <Tab label="Inference" />
        </Tabs>

        <TabPanel value={tabValue} index={0}>
          <ExperimentTable
            experiments={experiments}
            onStop={handleStopExperiment}
            getStatusColor={getStatusColor}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={1}>
          <ExperimentTable
            experiments={experiments.filter((e) => e.type === 'training')}
            onStop={handleStopExperiment}
            getStatusColor={getStatusColor}
          />
        </TabPanel>

        <TabPanel value={tabValue} index={2}>
          <ExperimentTable
            experiments={experiments.filter((e) => e.type === 'inference')}
            onStop={handleStopExperiment}
            getStatusColor={getStatusColor}
          />
        </TabPanel>
      </Card>

      {/* Training Dialog */}
      <Dialog open={openTrainingDialog} onClose={() => setOpenTrainingDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Start Training Experiment</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <TextField
              select
              label="Topology"
              value={trainingConfig.topologyId}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, topologyId: e.target.value })}
              fullWidth
            >
              {topologies.map((topo: any) => (
                <MenuItem key={topo.id} value={topo.id}>
                  {topo.name}
                </MenuItem>
              ))}
            </TextField>

            <TextField
              label="Episodes"
              type="number"
              value={trainingConfig.episodes}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, episodes: parseInt(e.target.value) })}
              fullWidth
            />

            <TextField
              label="Save Interval"
              type="number"
              value={trainingConfig.saveInterval}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, saveInterval: parseInt(e.target.value) })}
              fullWidth
            />

            <TextField
              label="Learning Rate"
              type="number"
              value={trainingConfig.learningRate}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, learningRate: parseFloat(e.target.value) })}
              fullWidth
              inputProps={{ step: 0.0001 }}
            />

            <TextField
              label="Batch Size"
              type="number"
              value={trainingConfig.batchSize}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, batchSize: parseInt(e.target.value) })}
              fullWidth
            />

            <TextField
              label="Gamma (Discount Factor)"
              type="number"
              value={trainingConfig.gamma}
              onChange={(e) => setTrainingConfig({ ...trainingConfig, gamma: parseFloat(e.target.value) })}
              fullWidth
              inputProps={{ step: 0.01, min: 0, max: 1 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenTrainingDialog(false)}>Cancel</Button>
          <Button onClick={handleStartTraining} variant="contained" disabled={loading}>
            Start Training
          </Button>
        </DialogActions>
      </Dialog>

      {/* Inference Dialog */}
      <Dialog open={openInferenceDialog} onClose={() => setOpenInferenceDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Start Inference Experiment</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <TextField
              select
              label="Topology"
              value={inferenceConfig.topologyId}
              onChange={(e) => setInferenceConfig({ ...inferenceConfig, topologyId: e.target.value })}
              fullWidth
            >
              {topologies.map((topo: any) => (
                <MenuItem key={topo.id} value={topo.id}>
                  {topo.name}
                </MenuItem>
              ))}
            </TextField>

            <TextField
              select
              label="Model Version"
              value={inferenceConfig.modelVersion}
              onChange={(e) => setInferenceConfig({ ...inferenceConfig, modelVersion: e.target.value })}
              fullWidth
            >
              <MenuItem value="best">Best Model</MenuItem>
              {models.map((model: any) => (
                <MenuItem key={model.id} value={model.version}>
                  {model.name} ({model.version})
                </MenuItem>
              ))}
            </TextField>

            <TextField
              label="Number of Flows"
              type="number"
              value={inferenceConfig.numFlows}
              onChange={(e) => setInferenceConfig({ ...inferenceConfig, numFlows: parseInt(e.target.value) })}
              fullWidth
            />

            <TextField
              label="Duration (steps)"
              type="number"
              value={inferenceConfig.duration}
              onChange={(e) => setInferenceConfig({ ...inferenceConfig, duration: parseInt(e.target.value) })}
              fullWidth
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenInferenceDialog(false)}>Cancel</Button>
          <Button onClick={handleStartInference} variant="contained" disabled={loading}>
            Start Inference
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
}

interface ExperimentTableProps {
  experiments: Experiment[];
  onStop: (experiment: Experiment) => void;
  getStatusColor: (status: string) => any;
}

function ExperimentRow({ experiment, onStop, getStatusColor }: { experiment: Experiment; onStop: (exp: Experiment) => void; getStatusColor: (status: string) => any }) {
  const [open, setOpen] = useState(false);
  const hasDetailedMetrics = experiment.type === 'inference' && experiment.metrics && 'qos' in experiment.metrics;

  return (
    <>
      <TableRow>
        <TableCell>
          {hasDetailedMetrics && (
            <IconButton size="small" onClick={() => setOpen(!open)}>
              {open ? <KeyboardArrowUp /> : <KeyboardArrowDown />}
            </IconButton>
          )}
        </TableCell>
        <TableCell>
          <Typography variant="body2" sx={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>
            {experiment.id}
          </Typography>
        </TableCell>
        <TableCell>
          <Chip label={experiment.type} size="small" />
        </TableCell>
        <TableCell>
          <Chip label={experiment.status} size="small" color={getStatusColor(experiment.status)} />
        </TableCell>
        <TableCell>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5, minWidth: 200 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <LinearProgress
                variant="determinate"
                value={experiment.progress || 0}
                sx={{ flexGrow: 1, height: 8, borderRadius: 4 }}
              />
              <Typography variant="body2" sx={{ minWidth: 45 }}>
                {Math.round(experiment.progress || 0)}%
              </Typography>
            </Box>
            {experiment.type === 'training' && experiment.currentEpisode !== undefined && (
              <Typography variant="caption" color="text.secondary">
                Episode {experiment.currentEpisode} / {experiment.totalEpisodes}
              </Typography>
            )}
          </Box>
        </TableCell>
        <TableCell>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
            {hasDetailedMetrics ? (
              <>
                <Typography variant="caption" color="text.secondary">
                  Avg Delay: {(experiment.metrics as any).qos?.avgDelay?.toFixed(2) || 0} ms
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  Packet Loss: {((experiment.metrics as any).qos?.packetLoss * 100)?.toFixed(2) || 0}%
                </Typography>
              </>
            ) : experiment.metrics ? (
              Object.entries(experiment.metrics).slice(0, 2).map(([key, value]) => (
                <Typography key={key} variant="caption" color="text.secondary">
                  {key}: {typeof value === 'number' ? value.toFixed(3) : value}
                </Typography>
              ))
            ) : null}
          </Box>
        </TableCell>
        <TableCell>
          <Typography variant="body2">
            {experiment.startTime ? new Date(experiment.startTime).toLocaleString() : '-'}
          </Typography>
        </TableCell>
        <TableCell>
          {experiment.status === 'running' && (
            <Button size="small" startIcon={<Stop />} onClick={() => onStop(experiment)} color="error">
              Stop
            </Button>
          )}
        </TableCell>
      </TableRow>
      {hasDetailedMetrics && (
        <TableRow>
          <TableCell style={{ paddingBottom: 0, paddingTop: 0 }} colSpan={8}>
            <Collapse in={open} timeout="auto" unmountOnExit>
              <Box sx={{ margin: 2 }}>
                <Typography variant="h6" gutterBottom component="div">
                  Detailed Metrics
                </Typography>
                <Grid container spacing={2}>
                  {/* QoS Metrics */}
                  <Grid item xs={12} md={6}>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="subtitle2" color="primary" gutterBottom>
                        QoS Metrics
                      </Typography>
                      <Divider sx={{ mb: 1 }} />
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                        <Typography variant="body2">
                          Avg End-to-End Delay: <strong>{(experiment.metrics as any).qos?.avgDelay?.toFixed(2) || 0} ms</strong>
                        </Typography>
                        <Typography variant="body2">
                          P95 Latency: <strong>{(experiment.metrics as any).qos?.p95Latency?.toFixed(2) || 0} ms</strong>
                        </Typography>
                        <Typography variant="body2">
                          P99 Latency: <strong>{(experiment.metrics as any).qos?.p99Latency?.toFixed(2) || 0} ms</strong>
                        </Typography>
                        <Typography variant="body2">
                          Packet Loss: <strong>{((experiment.metrics as any).qos?.packetLoss * 100)?.toFixed(2) || 0}%</strong>
                        </Typography>
                        <Typography variant="body2">
                          Jitter: <strong>{(experiment.metrics as any).qos?.jitter?.toFixed(2) || 0} ms</strong>
                        </Typography>
                        <Typography variant="body2">
                          SLA Violations: <strong>{((experiment.metrics as any).qos?.slaViolations * 100)?.toFixed(2) || 0}%</strong>
                        </Typography>
                      </Box>
                    </Paper>
                  </Grid>

                  {/* Network Utilization */}
                  <Grid item xs={12} md={6}>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="subtitle2" color="primary" gutterBottom>
                        Network Utilization
                      </Typography>
                      <Divider sx={{ mb: 1 }} />
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                        <Typography variant="body2">
                          Max Link Utilization: <strong>{(experiment.metrics as any).utilization?.maxLinkUtil?.toFixed(2) || 0}%</strong>
                        </Typography>
                        <Typography variant="body2">
                          Avg Link Utilization: <strong>{(experiment.metrics as any).utilization?.avgLinkUtil?.toFixed(2) || 0}%</strong>
                        </Typography>
                        <Typography variant="body2">
                          Fairness Index: <strong>{(experiment.metrics as any).utilization?.fairnessIndex?.toFixed(3) || 0}</strong>
                        </Typography>
                      </Box>
                    </Paper>
                  </Grid>

                  {/* Learning Metrics */}
                  <Grid item xs={12} md={6}>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="subtitle2" color="primary" gutterBottom>
                        Learning Metrics
                      </Typography>
                      <Divider sx={{ mb: 1 }} />
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                        <Typography variant="body2">
                          Convergence Speed: <strong>{((experiment.metrics as any).learning?.convergenceSpeed * 100)?.toFixed(1) || 0}%</strong>
                        </Typography>
                        <Typography variant="body2">
                          Policy Stability: <strong>{((experiment.metrics as any).learning?.policyStability * 100)?.toFixed(1) || 0}%</strong>
                        </Typography>
                        <Typography variant="body2">
                          Generalization: <strong>{((experiment.metrics as any).learning?.generalization * 100)?.toFixed(1) || 0}%</strong>
                        </Typography>
                      </Box>
                    </Paper>
                  </Grid>

                  {/* Operational Overhead */}
                  <Grid item xs={12} md={6}>
                    <Paper variant="outlined" sx={{ p: 2 }}>
                      <Typography variant="subtitle2" color="primary" gutterBottom>
                        Operational Overhead
                      </Typography>
                      <Divider sx={{ mb: 1 }} />
                      <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
                        <Typography variant="body2">
                          Routing Updates: <strong>{(experiment.metrics as any).overhead?.routingUpdates || 0}</strong>
                        </Typography>
                        <Typography variant="body2">
                          Control Traffic: <strong>{((experiment.metrics as any).overhead?.controlTraffic * 100)?.toFixed(2) || 0}%</strong>
                        </Typography>
                        <Typography variant="body2">
                          Adaptability: <strong>{((experiment.metrics as any).overhead?.adaptability * 100)?.toFixed(1) || 0}%</strong>
                        </Typography>
                      </Box>
                    </Paper>
                  </Grid>
                </Grid>
              </Box>
            </Collapse>
          </TableCell>
        </TableRow>
      )}
    </>
  );
}

function ExperimentTable({ experiments, onStop, getStatusColor }: ExperimentTableProps) {
  return (
    <TableContainer component={Paper} variant="outlined">
      <Table>
        <TableHead>
          <TableRow>
            <TableCell />
            <TableCell>ID</TableCell>
            <TableCell>Type</TableCell>
            <TableCell>Status</TableCell>
            <TableCell>Progress</TableCell>
            <TableCell>Quick Metrics</TableCell>
            <TableCell>Started</TableCell>
            <TableCell>Actions</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {experiments.length === 0 ? (
            <TableRow>
              <TableCell colSpan={8} align="center">
                <Typography variant="body2" color="text.secondary">
                  No experiments yet. Start a training or inference experiment to get started.
                </Typography>
              </TableCell>
            </TableRow>
          ) : (
            experiments.map((experiment) => (
              <ExperimentRow
                key={experiment.id}
                experiment={experiment}
                onStop={onStop}
                getStatusColor={getStatusColor}
              />
            ))
          )}
        </TableBody>
      </Table>
    </TableContainer>
  );
}
