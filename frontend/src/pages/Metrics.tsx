import { useEffect, useState } from 'react';
import {
  Typography,
  Box,
  Card,
  CardContent,
  Grid,
  Button,
  Paper,
} from '@mui/material';
import { Refresh, Download } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import ReactECharts from 'echarts-for-react';

export default function Metrics() {
  const dispatch = useAppDispatch();
  const experiments = useAppSelector((state) => state.experiments.items);

  // Mock data for demonstration
  const delayChartOption = {
    title: { text: 'Average Delay Over Time', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: ['0', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'] },
    yAxis: { type: 'value', name: 'Delay (ms)' },
    series: [
      {
        name: 'GNN-DRL',
        type: 'line',
        data: [45, 42, 38, 35, 33, 30, 28, 27, 26, 25, 24],
        smooth: true,
        itemStyle: { color: '#1976d2' },
      },
      {
        name: 'Shortest Path',
        type: 'line',
        data: [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
        smooth: true,
        itemStyle: { color: '#dc004e' },
      },
    ],
    legend: { data: ['GNN-DRL', 'Shortest Path'], bottom: 10 },
  };

  const rewardChartOption = {
    title: { text: 'Training Reward Progress', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: Array.from({ length: 21 }, (_, i) => (i * 5).toString()) },
    yAxis: { type: 'value', name: 'Reward' },
    series: [
      {
        name: 'Episode Reward',
        type: 'line',
        data: [-100, -80, -60, -50, -40, -30, -25, -20, -15, -12, -10, -8, -6, -5, -4, -3, -2, -1, 0, 2, 5],
        smooth: true,
        itemStyle: { color: '#2e7d32' },
      },
    ],
  };

  const utilizationChartOption = {
    title: { text: 'Link Utilization Distribution', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: ['Link 1', 'Link 2', 'Link 3', 'Link 4', 'Link 5', 'Link 6'] },
    yAxis: { type: 'value', name: 'Utilization (%)', max: 100 },
    series: [
      {
        name: 'Utilization',
        type: 'bar',
        data: [65, 45, 78, 52, 38, 70],
        itemStyle: { color: '#ed6c02' },
      },
    ],
  };

  const throughputChartOption = {
    title: { text: 'Throughput Comparison', left: 'center' },
    tooltip: { trigger: 'axis' },
    xAxis: { type: 'category', data: ['NSFNET', 'GEANT2', 'Custom 1', 'Custom 2'] },
    yAxis: { type: 'value', name: 'Throughput (Mbps)' },
    series: [
      {
        name: 'GNN-DRL',
        type: 'bar',
        data: [850, 920, 780, 890],
        itemStyle: { color: '#1976d2' },
      },
      {
        name: 'Baseline',
        type: 'bar',
        data: [650, 700, 600, 680],
        itemStyle: { color: '#dc004e' },
      },
    ],
    legend: { data: ['GNN-DRL', 'Baseline'], bottom: 10 },
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Performance Metrics</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button variant="outlined" startIcon={<Refresh />}>
            Refresh
          </Button>
          <Button variant="contained" startIcon={<Download />}>
            Export
          </Button>
        </Box>
      </Box>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Avg Delay
              </Typography>
              <Typography variant="h3" color="primary">
                24ms
              </Typography>
              <Typography variant="caption" color="success.main">
                ↓ 52% improvement
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
                850Mbps
              </Typography>
              <Typography variant="caption" color="success.main">
                ↑ 31% improvement
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
                0.8%
              </Typography>
              <Typography variant="caption" color="success.main">
                ↓ 65% improvement
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={3}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Avg Reward
              </Typography>
              <Typography variant="h3" color="primary">
                5.2
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Latest episode
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
            <ReactECharts option={throughputChartOption} style={{ height: '350px' }} />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}
