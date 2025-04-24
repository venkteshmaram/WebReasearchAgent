import React from 'react';
import Box from '@mui/material/Box';
import CircularProgress from '@mui/material/CircularProgress';
import Typography from '@mui/material/Typography';
import './LoadingIndicator.css';

const LoadingIndicator = () => (
  <Box className="loading-indicator-mui">
    <CircularProgress size={60} thickness={4} />
    <Typography variant="h6" sx={{ mt: 2, color: 'text.secondary' }}>
      Researching the web...
    </Typography>
    <Typography variant="body2" sx={{ mt: 1, color: 'text.secondary', maxWidth: 500, mx: 'auto' }}>
      Our AI agent is searching, crawling, and analyzing multiple sources to compile your research.
    </Typography>
  </Box>
);

export default LoadingIndicator; 