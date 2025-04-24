import React from 'react';
import Box from '@mui/material/Box';
import Typography from '@mui/material/Typography';
import Stack from '@mui/material/Stack';
import Link from '@mui/material/Link';
import SummarizeIcon from '@mui/icons-material/Summarize';
import Grow from '@mui/material/Grow';
import Skeleton from '@mui/material/Skeleton';
import Accordion from '@mui/material/Accordion';
import AccordionSummary from '@mui/material/AccordionSummary';
import AccordionDetails from '@mui/material/AccordionDetails';
import ExpandMoreIcon from '@mui/icons-material/ExpandMore';
import LinkIcon from '@mui/icons-material/Link';
import ReactMarkdown from 'react-markdown';
import './ResultsDisplay.css';

// Helper to safely get nested properties
const getNested = (obj, ...args) => {
  return args.reduce((obj, level) => obj && obj[level], obj);
};

const ResultsDisplay = ({ researchData, query, isLoading }) => { 
  // Tab state removed

  // Extract data
  const contentAnalysis = getNested(researchData, 'content_analysis');
  const synthesizedReport = getNested(researchData, 'synthesized_report');

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
  
  const SourceSkeleton = () => (
    <Accordion disabled sx={{ mb: 1.5 }}>
      <AccordionSummary expandIcon={<ExpandMoreIcon />}>
        <Skeleton width="60%" height={24} />
      </AccordionSummary>
      <AccordionDetails>
        <Skeleton />
        <Skeleton width="90%" />
      </AccordionDetails>
    </Accordion>
  );

  // SearchResultSkeleton removed

  return (
    <Box className="results-display-mui" sx={{ mt: 4 }}>
      {/* --- Synthesized Report Section (Directly Rendered) --- */}
      {isLoading && !synthesizedReport ? (
        <ReportSkeleton />
      ) : synthesizedReport ? (
         <Box sx={{ mb: 5 }}> 
           {/* Use ReactMarkdown to render the report */}
           <Box className="markdown-content" sx={{ fontSize: '1.1rem', lineHeight: 1.7, color: 'text.primary' }}>
              <ReactMarkdown>{synthesizedReport}</ReactMarkdown>
           </Box>
         </Box>
       ) : null}

      {/* --- Sources Section (Collapsible Accordions) --- */}
      {(isLoading || (contentAnalysis && Object.keys(contentAnalysis).length > 0)) && ( 
        <Box sx={{ mb: 4 }}>
           <Typography variant="h6" component="h2" gutterBottom sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
             <SummarizeIcon sx={{ mr: 1, color: 'info.main' }} /> Sources & Extracted Content
          </Typography>
          
          {/* Loading Skeletons */} 
          {isLoading && (!contentAnalysis || Object.keys(contentAnalysis).length === 0) ? (
             <Stack spacing={1.5}> {/* Reduced spacing */} 
               <SourceSkeleton />
               <SourceSkeleton />
               <SourceSkeleton />
             </Stack>
          )
          /* Render Accordions if analysis available */
          : contentAnalysis && Object.keys(contentAnalysis).length > 0 ? (
             Object.entries(contentAnalysis)
               // Render an accordion for items that are relevant AND have a summary
               .filter(([key, analysis]) => analysis && analysis.is_relevant && analysis.summary)
               .map(([key, analysis], index) => {
                  let sourceUrl = null;
                  let displaySourceText = 'Source'; // Default text
                  
                  if (key.startsWith('http')) { 
                    sourceUrl = key;
                    try { displaySourceText = new URL(sourceUrl).hostname; } catch { displaySourceText = "View Source"; }
                  } else if (key.startsWith('snippet::')) {
                    sourceUrl = key.substring(9); 
                    try { displaySourceText = new URL(sourceUrl).hostname; } catch { displaySourceText = "View Source"; }
                  }
                  // Identify non-URL sources for display difference
                  const isWebSource = !!sourceUrl; 

                  return (
                     <Grow in key={key} timeout={(index + 1) * 200}> 
                       <Accordion sx={{ mb: 1.5, 
                                        boxShadow: 'none', // Remove default shadow 
                                        border: 1, 
                                        borderColor: 'divider',
                                        '&:before': { display: 'none' } // Remove top border duplication
                                      }}
                       >
                         <AccordionSummary 
                            expandIcon={<ExpandMoreIcon />} 
                            aria-controls={`panel${index}-content`} 
                            id={`panel${index}-header`}
                            sx={{ '.MuiAccordionSummary-content': { alignItems: 'center' } }} // Align items in summary
                         >
                            {isWebSource ? (
                              <Link 
                                href={sourceUrl} 
                                target="_blank" 
                                rel="noopener noreferrer" 
                                underline="hover"
                                onClick={(e) => e.stopPropagation()} // Prevent accordion toggle on link click
                                sx={{ display: 'flex', alignItems: 'center', fontWeight: 500, mr: 1, flexGrow: 1 }} // Take available space
                              >
                                <LinkIcon sx={{ fontSize: '1.1rem', mr: 0.75, color: 'action.active' }} />
                                {displaySourceText}
                              </Link>
                            ) : (
                              <Typography sx={{ fontWeight: 500, mr: 1, flexGrow: 1, display: 'flex', alignItems: 'center' }}>
                                <SummarizeIcon sx={{ fontSize: '1.1rem', mr: 0.75, color: 'action.active' }} />
                                {key.startsWith('tavily_answer') ? 'Direct Answer' : 'Extracted Snippet'} 
                              </Typography>
                            )}
                         </AccordionSummary>
                         <AccordionDetails sx={{ pt: 0 }}> {/* Remove top padding */} 
                           <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                              {analysis.summary}
                           </Typography>
                         </AccordionDetails>
                       </Accordion>
                  </Grow>
                  );
               })
           )
           /* Message if no content was analyzed */
           : !isLoading ? ( 
             <Typography>No relevant source content was generated.</Typography>
           ) : null}
           
           {/* Message specifically if analysis ran but produced no relevant summaries */} 
           {!isLoading && contentAnalysis && Object.keys(contentAnalysis).length > 0 && Object.values(contentAnalysis).every(a => !a || !a.is_relevant || !a.summary) && 
              <Typography sx={{mt: 2, color: 'text.secondary'}}>Analysis complete, but no relevant summaries generated from the available content.</Typography>
            }
        </Box>
      )}

      {/* All Search Results Panel Removed */}

    </Box>
  );
};

export default ResultsDisplay; 