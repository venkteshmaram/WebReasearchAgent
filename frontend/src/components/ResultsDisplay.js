import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Link from '@mui/material/Link';
import SummarizeIcon from '@mui/icons-material/Summarize';
import Grow from '@mui/material/Grow';
import Skeleton from '@mui/material/Skeleton';
import LinkIcon from '@mui/icons-material/Link';
import List from '@mui/material/List';
import ListItem from '@mui/material/ListItem';
import ListItemText from '@mui/material/ListItemText';
import ListItemIcon from '@mui/material/ListItemIcon';
import ReactMarkdown from 'react-markdown';
import './ResultsDisplay.css';

const ResultsDisplay = ({ 
  isLoading,
  report,
  sources,
  queryAnalysis,
  searchResultsSummary,
  scrapingSummary
}) => { 
  // Tab state removed

  // --- Skeleton Components --- 
  const ReportSkeleton = () => (
    <Box sx={{ mb: 4 }}>
      <Typography variant="h5" component="h2" gutterBottom>
         <Skeleton width="40%" />
      </Typography>
      <Skeleton variant="rectangular" height={100} sx={{ mb: 1}} />
      <Skeleton />
      <Skeleton width="80%" />
      <Skeleton width="60%" />
    </Box>
  );
  
  const SourcesSkeleton = () => (
    <Box sx={{ mb: 4 }}>
       <Typography variant="h6" component="h2" gutterBottom sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
         <SummarizeIcon sx={{ mr: 1, color: 'info.main' }} /> Sources
      </Typography>
      <Stack spacing={1.5}>
         <Skeleton height={30} width="80%" />
         <Skeleton height={30} width="70%" />
         <Skeleton height={30} width="90%" />
      </Stack>
    </Box>
  );

  // SearchResultSkeleton removed

  // --- Helper function to render source links --- 
  const renderSourceLink = (source) => {
    let url = source; 
    let displayText = source;
    let isLink = false;

    if (source.startsWith('snippet::') || source.startsWith('http')) {
      isLink = true;
      url = source.startsWith('snippet::') ? source.substring(9) : source;
      try {
        displayText = new URL(url).hostname + (new URL(url).pathname === '/' ? '' : new URL(url).pathname);
      } catch (e) {
        displayText = url; // Fallback to full URL if parsing fails
      }
    } else if (source.startsWith('tavily_answer::')) {
       displayText = `Tavily Answer (${source.split('::')[1]})`;
    } else {
       displayText = source; // Fallback for unexpected formats
    }

    if (isLink) {
      return (
        <Link href={url} target="_blank" rel="noopener noreferrer" underline="hover" sx={{ wordBreak: 'break-all' }}>
          {displayText}
        </Link>
      );
    } else {
      return <Typography component="span" sx={{ wordBreak: 'break-all' }}>{displayText}</Typography>;
    }
  };
  // --- End Helper --- 

  return (
    <Box className="results-display-mui" sx={{ mt: 4 }}>
      {/* --- Synthesized Report Section (Directly Rendered) --- */}
      {isLoading && !report ? (
        <ReportSkeleton />
      ) : report ? (
         <Box sx={{ mb: 5 }}> 
           {/* Use ReactMarkdown to render the report */}
           <Box className="markdown-content" sx={{ fontSize: '1.1rem', lineHeight: 1.7, color: 'text.primary' }}>
              <ReactMarkdown>{report}</ReactMarkdown>
           </Box>
         </Box>
       ) : null}

      {/* --- NEW Sources Section --- */}
      {isLoading && (!sources || sources.length === 0) ? (
        <SourcesSkeleton />
      ) : sources && sources.length > 0 ? (
        <Box sx={{ mb: 4 }}>
           <Typography variant="h6" component="h2" gutterBottom sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
             <SummarizeIcon sx={{ mr: 1, color: 'info.main' }} /> Sources Used
          </Typography>
          <List dense>
            {sources.map((source, index) => (
              <Grow in key={`${source}-${index}`} timeout={(index + 1) * 150}> 
                <ListItem disableGutters>
                  <ListItemIcon sx={{ minWidth: '30px' }}> 
                     <LinkIcon fontSize="small" color="action" /> 
                  </ListItemIcon>
                  <ListItemText primary={renderSourceLink(source)} />
                </ListItem>
              </Grow>
            ))}
          </List>
        </Box>
      ) : null}
      {/* --- End Sources Section --- */} 

      {/* --- Optional: Display intermediate data if needed --- */}
      {/* 
      {queryAnalysis && ...}
      {searchResultsSummary && ...}
      {scrapingSummary && ...} 
      */}

    </Box>
  );
};

export default ResultsDisplay; 