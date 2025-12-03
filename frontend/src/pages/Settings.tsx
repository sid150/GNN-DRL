import { useState } from 'react';
import {
  Typography,
  Box,
  Card,
  CardContent,
  Grid,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import { Save, RestartAlt } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import { setTheme } from '@/store/slices/systemSlice';

export default function Settings() {
  const dispatch = useAppDispatch();
  const theme = useAppSelector((state) => state.system.theme);
  const [saved, setSaved] = useState(false);

  // General settings
  const [apiUrl, setApiUrl] = useState('http://localhost:8000');
  const [pollingInterval, setPollingInterval] = useState(5000);
  const [enableWebSocket, setEnableWebSocket] = useState(true);
  
  // Training settings
  const [defaultEpisodes, setDefaultEpisodes] = useState(100);
  const [defaultBatchSize, setDefaultBatchSize] = useState(32);
  const [defaultLearningRate, setDefaultLearningRate] = useState(0.001);
  const [defaultGamma, setDefaultGamma] = useState(0.99);
  
  // UI settings
  const [chartMaxPoints, setChartMaxPoints] = useState(1000);
  const [tablePageSize, setTablePageSize] = useState(25);
  const [enableAnimations, setEnableAnimations] = useState(true);

  const handleSave = () => {
    // Save settings logic would go here
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const handleReset = () => {
    setApiUrl('http://localhost:8000');
    setPollingInterval(5000);
    setEnableWebSocket(true);
    setDefaultEpisodes(100);
    setDefaultBatchSize(32);
    setDefaultLearningRate(0.001);
    setDefaultGamma(0.99);
    setChartMaxPoints(1000);
    setTablePageSize(25);
    setEnableAnimations(true);
    dispatch(setTheme('light'));
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Settings</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button variant="outlined" startIcon={<RestartAlt />} onClick={handleReset}>
            Reset to Defaults
          </Button>
          <Button variant="contained" startIcon={<Save />} onClick={handleSave}>
            Save Changes
          </Button>
        </Box>
      </Box>

      {saved && (
        <Alert severity="success" sx={{ mb: 3 }}>
          Settings saved successfully!
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* General Settings */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                General Settings
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="API Base URL"
                    value={apiUrl}
                    onChange={(e) => setApiUrl(e.target.value)}
                    helperText="Backend API server URL"
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Polling Interval (ms)"
                    type="number"
                    value={pollingInterval}
                    onChange={(e) => setPollingInterval(parseInt(e.target.value))}
                    helperText="How often to poll for updates"
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={enableWebSocket}
                        onChange={(e) => setEnableWebSocket(e.target.checked)}
                      />
                    }
                    label="Enable WebSocket for real-time updates"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* Training Defaults */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Training Default Values
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Default Episodes"
                    type="number"
                    value={defaultEpisodes}
                    onChange={(e) => setDefaultEpisodes(parseInt(e.target.value))}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Default Batch Size"
                    type="number"
                    value={defaultBatchSize}
                    onChange={(e) => setDefaultBatchSize(parseInt(e.target.value))}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Default Learning Rate"
                    type="number"
                    value={defaultLearningRate}
                    onChange={(e) => setDefaultLearningRate(parseFloat(e.target.value))}
                    inputProps={{ step: 0.0001 }}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Default Gamma"
                    type="number"
                    value={defaultGamma}
                    onChange={(e) => setDefaultGamma(parseFloat(e.target.value))}
                    inputProps={{ step: 0.01, min: 0, max: 1 }}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* UI Settings */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                User Interface
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <FormControl fullWidth>
                    <InputLabel>Theme</InputLabel>
                    <Select
                      value={theme}
                      label="Theme"
                      onChange={(e) => dispatch(setTheme(e.target.value as 'light' | 'dark'))}
                    >
                      <MenuItem value="light">Light</MenuItem>
                      <MenuItem value="dark">Dark</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Chart Max Data Points"
                    type="number"
                    value={chartMaxPoints}
                    onChange={(e) => setChartMaxPoints(parseInt(e.target.value))}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    fullWidth
                    label="Table Page Size"
                    type="number"
                    value={tablePageSize}
                    onChange={(e) => setTablePageSize(parseInt(e.target.value))}
                  />
                </Grid>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={enableAnimations}
                        onChange={(e) => setEnableAnimations(e.target.checked)}
                      />
                    }
                    label="Enable animations and transitions"
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>

        {/* System Information */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                System Information
              </Typography>
              <Divider sx={{ mb: 2 }} />
              <Grid container spacing={2}>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" color="text.secondary">
                    Frontend Version
                  </Typography>
                  <Typography variant="body1">1.0.0</Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" color="text.secondary">
                    API Status
                  </Typography>
                  <Typography variant="body1" color="success.main">
                    Connected
                  </Typography>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Typography variant="body2" color="text.secondary">
                    Environment
                  </Typography>
                  <Typography variant="body1">Development</Typography>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
}
