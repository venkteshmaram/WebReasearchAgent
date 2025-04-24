import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';
import Paper from '@mui/material/Paper';
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutline';
import './ErrorHandling.css';

const ErrorHandling = ({ error, setError }) => {
  if (!error) {
    return null;
  }

  const handleClearError = () => {
    setError(null);
  };
  
  const displayMessage = typeof error === 'string' && error ? error : "An unexpected error occurred.";

  return (
    <Paper 
      elevation={0}
      sx={{ 
        p: 3,
        textAlign: 'center',
        backgroundColor: (theme) => theme.palette.mode === 'light' ? '#fff8f8' : '#2d1f1f',
        border: (theme) => `1px solid ${theme.palette.error.light}`,
        color: (theme) => theme.palette.error.dark,
        borderRadius: 2,
        my: 3
      }}
    >
      <Box sx={{ color: 'error.main', mb: 1.5 }}>
        <ErrorOutlineIcon fontSize="large" />
      </Box>
      
      <Typography 
        variant="body1"
        component="p" 
        gutterBottom
        sx={{ fontWeight: 500, mb: 1 }}
      >
        {displayMessage} 
      </Typography>
      
      <Typography 
        variant="body2" 
        color="text.secondary"
        paragraph
        sx={{ maxWidth: 500, mx: 'auto', mb: 2.5 }}
      >
        We encountered an issue while processing your request. Please try again or modify your search query.
      </Typography>
      
      <Button
        variant="outlined"
        color="error"
        onClick={handleClearError}
        size="small"
      >
        Dismiss
      </Button>
    </Paper>
  );
};

export default ErrorHandling; 