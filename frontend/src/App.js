import React, { useState, useMemo, useEffect, useRef } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Container from '@mui/material/Container';
import Box from '@mui/material/Box';
import AppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import Typography from '@mui/material/Typography';
import SearchIcon from '@mui/icons-material/Search';
import IconButton from '@mui/material/IconButton';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import LinearProgress from '@mui/material/LinearProgress';
import './App.css';
import SearchBar from './components/SearchBar';
import ResultsDisplay from './components/ResultsDisplay';
import ErrorHandling from './components/ErrorHandling';
import Footer from './components/Footer';
import Sidebar from './components/Sidebar';

function App() {
  const [mode, setMode] = useState('light');
  const [query, setQuery] = useState('');
  const [researchData, setResearchData] = useState(null);
  const [isLoading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentStatus, setCurrentStatus] = useState('');
  const [progress, setProgress] = useState(0);
  const eventSourceRef = useRef(null);
  const [researchCompletedSuccessfully, setResearchCompletedSuccessfully] = useState(false);
  const [recentSearches, setRecentSearches] = useState([]);

  const theme = useMemo(() => createTheme({
    palette: {
      mode,
      ...(mode === 'light' 
        ? { // Light Mode Palette
            primary: {
              main: '#212121', // Dark Gray/Black
            },
            secondary: {
              main: '#757575', // Medium Gray
            },
            background: {
              default: '#F5F5F5', // Very Light Gray
              paper: '#FFFFFF',   // White
            },
            text: {
              primary: '#212121', // Dark Gray/Black
              secondary: '#757575', // Medium Gray
            }
          }
        : { // Dark Mode Palette
            primary: {
              main: '#FFFFFF',   // White
            },
            secondary: {
              main: '#BDBDBD', // Light Gray
            },
            background: {
              default: '#121212', // Very Dark Gray/Black
              paper: '#1E1E1E',   // Dark Gray
            },
            text: {
              primary: '#FFFFFF',   // White
              secondary: '#BDBDBD', // Light Gray
            }
          }
      ),
      divider: mode === 'light' ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.12)',
      action: {
          active: mode === 'light' ? 'rgba(0, 0, 0, 0.54)' : 'rgba(255, 255, 255, 0.7)',
      },
    },
    shape: { borderRadius: 8 },
    typography: {
      fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
      h1: { fontSize: '2.4rem', fontWeight: 600, marginBottom: '0.5em' },
      h2: { fontSize: '1.9rem', fontWeight: 600, marginBottom: '0.5em' },
      h3: { fontSize: '1.6rem', fontWeight: 500 },
      h4: { fontSize: '1.4rem', fontWeight: 500 },
      h5: { fontSize: '1.2rem', fontWeight: 500 },
      h6: { fontSize: '1.0rem', fontWeight: 500 },
      subtitle1: { fontSize: '1.1rem', color: 'text.secondary' },
      body1: { fontSize: '1rem' },
      body2: { fontSize: '0.9rem' },
      button: {
        textTransform: 'none',
        fontWeight: 600,
      }
    },
    components: {
      MuiButton: {
        styleOverrides: {
          root: {
            borderRadius: 8,
          },
        },
      },
      MuiCard: {
        styleOverrides: {
          root: {
            borderRadius: 8,
            boxShadow: 'none',
            border: `1px solid ${mode === 'light' ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.12)'}`, 
          }
        }
      },
      MuiAppBar: {
         styleOverrides: {
           root: {
             background: mode === 'light' ? '#FFFFFF' : '#1E1E1E',
             color: mode === 'light' ? '#212121' : '#FFFFFF', 
             boxShadow: 'none',
             borderBottom: `1px solid ${mode === 'light' ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.12)'}`, 
           }
         }
      },
      MuiAccordion: {
        styleOverrides: {
          root: {
            boxShadow: 'none',
            border: `1px solid ${mode === 'light' ? 'rgba(0, 0, 0, 0.12)' : 'rgba(255, 255, 255, 0.12)'}`, 
            borderRadius: 8,
            '&:before': { display: 'none' },
            backgroundColor: mode === 'light' ? '#FFFFFF' : '#1E1E1E',
          }
        }
      },
      MuiAccordionSummary: {
        styleOverrides: {
          root: {
            paddingLeft: '16px',
            paddingRight: '8px' 
          }
        }
      }
    }
  }), [mode]);

  const closeEventSource = () => {
    if (eventSourceRef.current) {
      console.log("Closing existing EventSource connection.");
      eventSourceRef.current.close();
      eventSourceRef.current = null;
      setLoading(false);
    }
  };

  useEffect(() => {
    return () => {
      closeEventSource();
    };
  }, []);

  const fetchResults = (searchQuery) => {
    const trimmedQuery = searchQuery.trim();
    if (!trimmedQuery) return;

    closeEventSource();
    setLoading(true);
    setError(null);
    setResearchData(null);
    setCurrentStatus('Agent started research...');
    setProgress(0);
    setResearchCompletedSuccessfully(false);

    if (!recentSearches.includes(trimmedQuery)) {
      setRecentSearches(prev => [trimmedQuery, ...prev].slice(0, 10));
    }

    const encodedQuery = encodeURIComponent(trimmedQuery);
    
    // --- Use full backend URL in production, relative path locally --- 
    const backendHost = process.env.REACT_APP_BACKEND_URL || ''; // Get from env var or default to empty string
    const url = `${backendHost}/api/research-stream?query=${encodedQuery}`;
    // --- End URL construction ---
    
    console.log(`Connecting to SSE endpoint: ${url}`);
    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.onopen = () => {
      console.log("SSE connection opened.");
      setCurrentStatus('Connection established, starting research...');
    };

    es.onmessage = (event) => {
      setLoading(true);
      console.log("SSE message received:", event.data);
      try {
        const parsedData = JSON.parse(event.data);
        
        if (parsedData.message) setCurrentStatus(parsedData.message);
        if (parsedData.progress) setProgress(parsedData.progress);

        if (parsedData.data) {
          setResearchData(prevData => ({
            ...prevData,
            ...parsedData.data,
          }));
        }

        if (parsedData.status === 'synthesizing_report_done') {
          console.log("Research complete (final message received).");
          setCurrentStatus("Research complete!");
          setProgress(1);
          setResearchCompletedSuccessfully(true);
          closeEventSource();
        } else if (parsedData.status === 'error') {
          console.error("Backend error received:", parsedData.message);
          setError(parsedData.message || 'An error occurred during research.');
          setCurrentStatus("Research failed.");
          setProgress(0);
          closeEventSource();
        }
      } catch (e) {
        console.error("Failed to parse SSE message:", event.data, e);
        setError('Received malformed data from server.');
        closeEventSource();
      }
    };

    es.onerror = (err) => {
      // Check if the EventSource is already closed or closing
      if (es.readyState === EventSource.CLOSED) {
        console.log("onerror triggered, but EventSource is already closed.");
        // If research completed successfully, ensure error state is null
        if (researchCompletedSuccessfully) {
           setError(null); // Explicitly clear error if success flag is true
        }
        closeEventSource(); // Ensure cleanup
        return; // Don't proceed further
      }

      // Existing check: Only set error if research wasn't marked complete
      if (!researchCompletedSuccessfully && eventSourceRef.current) {
        console.error("EventSource error before completion:", err);
        // Log the state of researchCompletedSuccessfully right before setting error
        console.log(`Error condition met. researchCompletedSuccessfully=${researchCompletedSuccessfully}`); 
        setError('Connection to server failed or was lost during research.');
        setCurrentStatus('Connection failed.');
        setProgress(0);
      } else {
         console.log("onerror triggered, but research already completed or ref is null.");
         // Also ensure error is cleared if success flag is true here
         if (researchCompletedSuccessfully) {
             setError(null);
         }
      }
      // Always attempt to close the source on error
      closeEventSource();
    };
  };

  const handleSearch = () => {
    fetchResults(query);
  };

  const handleSelectQuery = (selectedQuery) => {
    setQuery(selectedQuery);
    fetchResults(selectedQuery);
  };

  const handleClear = () => {
    closeEventSource();
    setQuery('');
    setResearchData(null);
    setError(null);
    setCurrentStatus('');
    setProgress(0);
    setLoading(false);
  };

  const isRunning = !!eventSourceRef.current;

  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <Box sx={{ display: 'flex' }}>
        {/* Sidebar */}
        <Sidebar 
            recentSearches={recentSearches} 
            onSelectQuery={handleSelectQuery}
            isRunning={isRunning}
        />
        
        {/* Main Content */}
        <Box component="main" sx={{ flexGrow: 1, p: 3 }}>
          {/* AppBar */}
          <AppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
            <Toolbar>
              <SearchIcon sx={{ mr: 1.5 }} />
              <Typography variant="h6" component="div" sx={{ flexGrow: 1 }}>
                Web Research Agent
              </Typography>
              <IconButton sx={{ ml: 1 }} onClick={() => setMode(mode === 'light' ? 'dark' : 'light')} color="inherit">
                {mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
              </IconButton>
            </Toolbar>
             {/* Progress Bar */}
             {isLoading && (
                 <LinearProgress variant={progress === 0 || progress === 1 ? 'indeterminate' : 'determinate'} value={progress * 100} />
             )}
          </AppBar>

          {/* Content Area */}
          <Container maxWidth="lg" sx={{ mt: 10 }}> {/* Adjusted margin top for fixed AppBar */}
            
            {/* Only show intro text if no research data and not loading */}
            {!researchData && !isLoading && (
              <Box sx={{ textAlign: 'center', my: 8 }}>
                <Typography variant="h3" component="h1" gutterBottom color="text.primary"> {/* Explicit color */}
                  AI-Powered Web Research
                </Typography>
                <Typography variant="subtitle1" color="text.secondary"> {/* Explicit color (already uses secondary, but good to be explicit) */}
                  Enter a query and let our agent synthesize information from the web.
                </Typography>
              </Box>
            )}

            {/* Search Bar */}
            <SearchBar
              query={query}
              onQueryChange={setQuery}
              onSearch={handleSearch}
              isLoading={isLoading}
              handleClear={handleClear}
            />

            {/* Status Message */}
            {isLoading && currentStatus && (
              <Typography sx={{ mt: 2, mb: 1, textAlign: 'center' }} color="text.primary"> {/* Explicit color */}
                {currentStatus}
              </Typography>
            )}
            
            {/* Error Display */}
            <ErrorHandling error={error} setError={setError} />

            {/* Results Display */}
            {(researchData || isLoading) && (
              <ResultsDisplay 
                researchData={researchData} 
                query={query} 
                isLoading={isLoading} 
              />
            )}
            
          </Container>
          
          {/* Footer */}
          <Footer />
        </Box>
      </Box>
    </ThemeProvider>
  );
}

export default App;
