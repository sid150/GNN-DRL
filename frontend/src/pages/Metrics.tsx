import { useEffect, useState } from 'react';
import {
  Typography,
  Box,
  Card,
  CardContent,
  Grid,
  Button,
  Paper,
  CircularProgress,
  Alert,
} from '@mui/material';
import { Refresh, Download } from '@mui/icons-material';
import ReactECharts from 'echarts-for-react';
import { api } from '@/services/api';

interface MetricsData {
  timestamp: string;
  learning: any[];
  utilization: any[];
  qos_summary: any;
  current_episode: number;
}

export default function Metrics() {
  const [metricsData, setMetricsData] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      setError(null);
      const history = await api.metrics.getHistory();
      setMetricsData(history);
    } catch (err: any) {
      setError(err.message || 'Failed to fetch metrics');
      console.error('Error fetching metrics:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    // Refresh every 5 seconds
    const interval = setInterval(fetchMetrics, 5000);
    return () => clearInterval(interval);
  }, []);

  const handleRefresh = () => {
    fetchMetrics();
  };

  const handleExport = async () => {
    try {
      // TODO: Implement export functionality
      console.log('Export metrics');
    } catch (err) {
      console.error('Error exporting metrics:', err);
    }
  };

  if (loading && !metricsData) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mb: 3 }}>
        {error}
      </Alert>
    );
  }

  // Extract data for charts
  const learningHistory = metricsData?.learning || [];
  const utilizationHistory = metricsData?.utilization || [];
  const qosSummary = metricsData?.qos_summary || {};

  // Prepare reward chart data
  const rewardData = learningHistory.map((item: any) => item.avg_reward || 0);
  const episodeLabels = learningHistory.map((_: any, idx: number) => idx.toString());

  // Prepare utilization chart data
  const avgUtilization = utilizationHistory.map((item: any) => item.avg_link_utilization || 0);
  const maxUtilization = utilizationHistory.map((item: any) => item.max_link_utilization || 0);

  // Current metrics
  const avgLatency = qosSummary.avg_latency_ms || 0;
  const avgThroughput = qosSummary.avg_throughput_mbps || 0;
  const avgPacketLoss = (qosSummary.avg_packet_loss || 0) * 100;
  const lastReward = rewardData.length > 0 ? rewardData[rewardData.length - 1] : 0;

  // Chart options
  const rewardChartOption = {
    title: { text: 'Training Reward Progress', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { 
      type: 'category', 
      data: episodeLabels.length > 0 ? episodeLabels : ['No data'] 
    },
    yAxis: { type: 'value', name: 'Reward' },
    series: [
      {
        name: 'Episode Reward',
        type: 'line',
        data: rewardData.length > 0 ? rewardData : [0],
        smooth: true,
        itemStyle: { color: '#2e7d32' },
      },
    ],
  };

  const utilizationChartOption = {
    title: { text: 'Link Utilization Over Time', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { 
      type: 'category', 
      data: utilizationHistory.map((_: any, idx: number) => idx.toString())
    },
    yAxis: { type: 'value', name: 'Utilization (%)', max: 100 },
    series: [
      {
        name: 'Avg Utilization',
        type: 'line',
        data: avgUtilization,
        smooth: true,
        itemStyle: { color: '#1976d2' },
      },
      {
        name: 'Max Utilization',
        type: 'line',
        data: maxUtilization,
        smooth: true,
        itemStyle: { color: '#dc004e' },
      },
    ],
    legend: { data: ['Avg Utilization', 'Max Utilization'], bottom: 10 },
  };

  const fairnessChartOption = {
    title: { text: 'Network Fairness Index', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { 
      type: 'category', 
      data: utilizationHistory.map((_: any, idx: number) => idx.toString())
    },
    yAxis: { type: 'value', name: 'Fairness Index', max: 1 },
    series: [
      {
        name: 'Jain\'s Fairness',
        type: 'line',
        data: utilizationHistory.map((item: any) => item.j_fairness || 0),
        smooth: true,
        itemStyle: { color: '#ed6c02' },
      },
    ],
  };

  const delayChartOption = {
    title: { text: 'Average Latency Trend', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { 
      type: 'category', 
      data: learningHistory.map((_: any, idx: number) => idx.toString())
    },
    yAxis: { type: 'value', name: 'Latency (ms)' },
    series: [
      {
        name: 'Avg Latency',
        type: 'line',
        data: learningHistory.map(() => avgLatency),
        smooth: true,
        itemStyle: { color: '#1976d2' },
      },
    ],
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Performance Metrics</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button variant="outlined" startIcon={<Refresh />} onClick={handleRefresh}>
            Refresh
          </Button>
          <Button variant="contained" startIcon={<Download />} onClick={handleExport}>
            Export
          </Button>
        </Box>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Avg Latency
              </Typography>
              <Typography variant="h3" color="primary">
                {avgLatency.toFixed(2)}ms
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Current average
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Throughput
              </Typography>
              <Typography variant="h3" color="primary">
                {avgThroughput.toFixed(0)}Mbps
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Current average
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Packet Loss
              </Typography>
              <Typography variant="h3" color="primary">
                {avgPacketLoss.toFixed(2)}%
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Current rate
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Last Reward
              </Typography>
              <Typography variant="h3" color="primary">
                {lastReward.toFixed(2)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Episode {metricsData?.current_episode || 0}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <ReactECharts option={delayChartOption} style={{ height: '350px' }} />
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <ReactECharts option={rewardChartOption} style={{ height: '350px' }} />
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <ReactECharts option={utilizationChartOption} style={{ height: '350px' }} />
          </Paper>
        </Grid>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 2 }}>
            <ReactECharts option={fairnessChartOption} style={{ height: '350px' }} />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
