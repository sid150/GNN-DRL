import { useEffect } from 'react';
import {
  Typography,
  Box,
  Card,
  CardContent,
  Grid,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Chip,
  IconButton,
  Tooltip,
  LinearProgress,
} from '@mui/material';
import { Refresh, Download, Delete, CheckCircle, RadioButtonUnchecked } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import { fetchModels } from '@/store/slices/modelsSlice';

export default function Models() {
  const dispatch = useAppDispatch();
  const { items: models, loading, activeModelId } = useAppSelector((state) => state.models);

  useEffect(() => {
    dispatch(fetchModels());
  }, [dispatch]);

  const handleRefresh = () => {
    dispatch(fetchModels());
  };

  const activeModel = models.find((m) => m.id === activeModelId);

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Model Versions</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button variant="outlined" startIcon={<Refresh />} onClick={handleRefresh}>
            Refresh
          </Button>
        </Box>
      </Box>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Total Models
              </Typography>
              <Typography variant="h3">{models.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Active Model
              </Typography>
              <Typography variant="h5">
                {activeModel ? activeModel.name : 'None'}
              </Typography>
              {activeModel && (
                <Typography variant="caption" color="text.secondary">
                  Version: {activeModel.version}
                </Typography>
              )}
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Best Performance
              </Typography>
              <Typography variant="h5">
                {activeModel?.metrics?.avgReward?.toFixed(3) || 'N/A'}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Average Reward
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Card>
        <CardContent>
          <TableContainer component={Paper} variant="outlined">
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Status</TableCell>
                  <TableCell>Version</TableCell>
                  <TableCell>Name</TableCell>
                  <TableCell>Performance</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell>Checkpoint Path</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {models.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={7} align="center">
                      <Typography variant="body2" color="text.secondary" sx={{ py: 3 }}>
                        {loading ? 'Loading models...' : 'No models available. Train a model to get started.'}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  models.map((model) => (
                    <TableRow key={model.id}>
                      <TableCell>
                        {model.active ? (
                          <Tooltip title="Active Model">
                            <CheckCircle color="success" />
                          </Tooltip>
                        ) : (
                          <RadioButtonUnchecked color="disabled" />
                        )}
                      </TableCell>
                      <TableCell>
                        <Chip label={model.version} size="small" />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body1" fontWeight="medium">
                          {model.name}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ minWidth: 200 }}>
                          {model.metrics && (
                            <>
                              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                                <Typography variant="caption" sx={{ minWidth: 80 }}>
                                  Avg Reward:
                                </Typography>
                                <LinearProgress
                                  variant="determinate"
                                  value={Math.min(model.metrics.avgReward * 10, 100)}
                                  sx={{ flexGrow: 1, height: 6, borderRadius: 3 }}
                                />
                                <Typography variant="caption">
                                  {model.metrics.avgReward?.toFixed(2)}
                                </Typography>
                              </Box>
                              <Typography variant="caption" color="text.secondary">
                                Accuracy: {model.metrics.accuracy?.toFixed(2)}
                              </Typography>
                            </>
                          )}
                        </Box>
                      </TableCell>
                      <TableCell>{new Date(model.createdAt).toLocaleDateString()}</TableCell>
                      <TableCell>
                        <Typography variant="caption" sx={{ fontFamily: 'monospace', fontSize: '0.75rem' }}>
                          {model.checkpointPath}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Tooltip title="Download">
                            <IconButton size="small" disabled>
                              <Download fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton size="small" disabled={model.active} color="error">
                              <Delete fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      </TableCell>
                    </TableRow>
                  ))
                )}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>
    </Box>
  );
}
