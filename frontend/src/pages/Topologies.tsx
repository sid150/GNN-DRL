import { useEffect, useState } from 'react';
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
} from '@mui/material';
import { Refresh, Visibility, Download, Delete, Add } from '@mui/icons-material';
import { useAppDispatch, useAppSelector } from '@/store/hooks';
import { fetchTopologies } from '@/store/slices/topologiesSlice';

export default function Topologies() {
  const dispatch = useAppDispatch();
  const { items: topologies, loading } = useAppSelector((state) => state.topologies);

  useEffect(() => {
    dispatch(fetchTopologies());
  }, [dispatch]);

  const handleRefresh = () => {
    dispatch(fetchTopologies());
  };

  return (
    <Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4">Network Topologies</Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Button variant="outlined" startIcon={<Refresh />} onClick={handleRefresh}>
            Refresh
          </Button>
          <Button variant="contained" startIcon={<Add />} disabled>
            Import Topology
          </Button>
        </Box>
      </Box>

      <Grid container spacing={3} sx={{ mb: 3 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Total Topologies
              </Typography>
              <Typography variant="h3">{topologies.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Total Nodes
              </Typography>
              <Typography variant="h3">
                {topologies.reduce((sum, t) => sum + t.nodes.length, 0)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Total Links
              </Typography>
              <Typography variant="h3">
                {topologies.reduce((sum, t) => sum + t.links.length, 0)}
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
                  <TableCell>ID</TableCell>
                  <TableCell>Name</TableCell>
                  <TableCell>Nodes</TableCell>
                  <TableCell>Links</TableCell>
                  <TableCell>Created</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {topologies.length === 0 ? (
                  <TableRow>
                    <TableCell colSpan={6} align="center">
                      <Typography variant="body2" color="text.secondary" sx={{ py: 3 }}>
                        {loading ? 'Loading topologies...' : 'No topologies available. Import a topology to get started.'}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ) : (
                  topologies.map((topology) => (
                    <TableRow key={topology.id}>
                      <TableCell>
                        <Chip label={topology.id} size="small" />
                      </TableCell>
                      <TableCell>
                        <Typography variant="body1" fontWeight="medium">
                          {topology.name}
                        </Typography>
                      </TableCell>
                      <TableCell>{topology.nodes.length}</TableCell>
                      <TableCell>{topology.links.length}</TableCell>
                      <TableCell>{new Date(topology.createdAt).toLocaleDateString()}</TableCell>
                      <TableCell>
                        <Box sx={{ display: 'flex', gap: 1 }}>
                          <Tooltip title="View Details">
                            <IconButton size="small" disabled>
                              <Visibility fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Export">
                            <IconButton size="small" disabled>
                              <Download fontSize="small" />
                            </IconButton>
                          </Tooltip>
                          <Tooltip title="Delete">
                            <IconButton size="small" disabled color="error">
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
