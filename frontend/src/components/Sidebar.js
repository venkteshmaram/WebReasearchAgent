import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemText from '@mui/material/ListItemText';
import Divider from '@mui/material/Divider';
import Chip from '@mui/material/Chip';
import HistoryIcon from '@mui/icons-material/History';
import LightbulbIcon from '@mui/icons-material/Lightbulb';

// Example queries - these might be passed as props later
const EXAMPLE_QUERIES = [
  'What are the latest advancements in AI?',
  'Compare React vs Angular in 2024',
  'Best practices for machine learning',
  'Climate change impact on agriculture'
];

const Sidebar = ({ recentSearches = [], onSelectQuery }) => {

  const handleSelect = (query) => {
    if (onSelectQuery) {
      onSelectQuery(query);
    }
  };

  return (
    <Box 
      sx={{ 
        width: 280, // Fixed width for the sidebar
        flexShrink: 0, 
        height: '100vh', // Let height be 100vh
        borderRight: 1, 
        borderColor: 'divider',
        p: 2,
        pt: 8, // Add padding top to account for AppBar (64px)
        overflowY: 'auto', // Allow scrolling if content overflows
        bgcolor: 'background.paper' // Ensure background color
      }}
    >
      {/* Recent Searches Section */}
      <Box mb={3}>
        <Typography variant="overline" display="flex" alignItems="center" sx={{ mb: 1, color: 'text.secondary' }}>
          <HistoryIcon sx={{ mr: 1, fontSize: '1.1rem' }} /> Recent Searches
        </Typography>
        {recentSearches.length > 0 ? (
          <List dense disablePadding sx={{ mt: 2 }}>
            {recentSearches.map((search, index) => (
              <ListItem key={index} disablePadding>
                <ListItemButton onClick={() => handleSelect(search)} dense>
                  <ListItemText 
                    primary={search} 
                    primaryTypographyProps={{ 
                      fontSize: '0.875rem', 
                      whiteSpace: 'nowrap', 
                      overflow: 'hidden', 
                      textOverflow: 'ellipsis' 
                    }} 
                  />
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        ) : (
          <Typography variant="caption" color="text.secondary">No recent searches.</Typography>
        )}
      </Box>

      <Divider sx={{ mb: 3 }} />

      {/* Example Queries Section */}
      <Box>
        <Typography variant="overline" display="flex" alignItems="center" sx={{ mb: 1, color: 'text.secondary' }}>
           <LightbulbIcon sx={{ mr: 1, fontSize: '1.1rem' }} /> Try Asking
        </Typography>
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
          {EXAMPLE_QUERIES.map((example, index) => (
            <Chip
              key={index}
              label={example}
              onClick={() => handleSelect(example)}
              variant="outlined"
              size="small"
              sx={{ cursor: 'pointer' }}
            />
          ))}
        </Box>
      </Box>

    </Box>
  );
};

export default Sidebar; 