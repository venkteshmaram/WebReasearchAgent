import React from 'react';
import Box from '@mui/material/Box';
import Container from '@mui/material/Container';
import Typography from '@mui/material/Typography';
import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import Divider from '@mui/material/Divider';
import GitHubIcon from '@mui/icons-material/GitHub';
import InfoIcon from '@mui/icons-material/Info';
import ContactSupportIcon from '@mui/icons-material/ContactSupport';
import PrivacyTipIcon from '@mui/icons-material/PrivacyTip';
import './Footer.css';

const Footer = () => (
  <Box
    component="footer"
    sx={{
      py: 3,
      px: 2,
      mt: 'auto',
      backgroundColor: (theme) => theme.palette.grey[100],
    }}
  >
    <Container maxWidth="md">
      <Stack
        direction="row"
        spacing={2}
        divider={<Divider orientation="vertical" flexItem />}
        justifyContent="center"
        sx={{ mb: 3 }}
      >
        <Link href="/about" underline="none" className="footer-link">
          <Stack direction="row" spacing={1} alignItems="center">
            <InfoIcon fontSize="small" />
            <Typography variant="body2">About</Typography>
          </Stack>
        </Link>
        
        <Link href="/contact" underline="none" className="footer-link">
          <Stack direction="row" spacing={1} alignItems="center">
            <ContactSupportIcon fontSize="small" />
            <Typography variant="body2">Contact</Typography>
          </Stack>
        </Link>
        
        <Link href="/privacy" underline="none" className="footer-link">
          <Stack direction="row" spacing={1} alignItems="center">
            <PrivacyTipIcon fontSize="small" />
            <Typography variant="body2">Privacy</Typography>
          </Stack>
        </Link>
        
        <Link href="https://github.com" target="_blank" rel="noopener noreferrer" underline="none" className="footer-link">
          <Stack direction="row" spacing={1} alignItems="center">
            <GitHubIcon fontSize="small" />
            <Typography variant="body2">GitHub</Typography>
          </Stack>
        </Link>
      </Stack>
      
      <Typography variant="body2" color="text.secondary" align="center">
        &copy; {new Date().getFullYear()} Web Research Agent. All rights reserved.
      </Typography>
      
      <Typography variant="caption" color="text.secondary" align="center" sx={{ mt: 1, display: 'block' }}>
        Powered by AI and built with Modern React
      </Typography>
    </Container>
  </Box>
);

export default Footer; 