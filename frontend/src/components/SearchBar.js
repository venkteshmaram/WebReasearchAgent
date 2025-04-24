import React from 'react';
import Paper from '@mui/material/Paper';
import InputBase from '@mui/material/InputBase';
import IconButton from '@mui/material/IconButton';
import Button from '@mui/material/Button';
import Divider from '@mui/material/Divider';
import SearchIcon from '@mui/icons-material/Search';
import ClearIcon from '@mui/icons-material/Clear';
import './SearchBar.css';

const SearchBar = ({ query, onQueryChange, onSearch, onClear, disabled, currentStatus }) => {
  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !disabled) {
      onSearch();
    }
  };

  const handleSearch = () => {
    if (query.trim() && !disabled) {
      onSearch();
    }
  };

  // Determine button text based on disabled state and status
  let buttonText = "Search";
  if (disabled && currentStatus) {
      switch (currentStatus) {
          case 'analyzing_query':
              buttonText = "Analyzing...";
              break;
          case 'searching_web':
              buttonText = "Searching...";
              break;
          case 'scraping_pages':
              buttonText = "Scraping...";
              break;
          case 'analyzing_content':
              buttonText = "Analyzing Content...";
              break;
          case 'synthesizing_report':
              buttonText = "Synthesizing...";
              break;
          case 'complete':
              buttonText = "Done!";
              break;
          case 'error':
              buttonText = "Error";
              break;
          default:
              if (currentStatus.endsWith('_done')) {
                  buttonText = "Processing...";
              } else {
                 buttonText = "Processing...";
              }
              break;
      }
  }

  return (
    <Paper
      component="form"
      onSubmit={(e) => {
        e.preventDefault();
        handleSearch();
      }}
      elevation={2}
      sx={{
        p: '8px 12px',
        display: 'flex',
        alignItems: 'center',
        borderRadius: 3,
        border: (theme) => `1px solid ${theme.palette.divider}`,
        backgroundColor: 'background.paper',
        width: '100%'
      }}
    >
      <InputBase
        sx={{ ml: 1, flex: 1 }}
        placeholder="Enter your research query..."
        value={query}
        onChange={(e) => onQueryChange(e.target.value)}
        onKeyPress={handleKeyPress}
        inputProps={{ 'aria-label': 'search web' }}
        disabled={disabled}
      />
      {query && (
        <IconButton 
          aria-label="clear" 
          onClick={onClear}
          sx={{ color: 'rgba(0, 0, 0, 0.54)' }}
          disabled={disabled}
        >
          <ClearIcon />
        </IconButton>
      )}
      <Divider sx={{ height: 28, m: 0.5 }} orientation="vertical" />
      <Button 
        onClick={handleSearch} 
        variant="contained"
        disableElevation
        startIcon={<SearchIcon />}
        disabled={disabled}
        sx={{
          borderRadius: 2,
          ml: 1,
          px: 3,
          height: 44,
          minWidth: '130px',
          fontWeight: 'bold',
        }}
      >
        {buttonText}
      </Button>
    </Paper>
  );
};

export default SearchBar; 