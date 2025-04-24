# Web Research Agent

This project implements an AI-powered Web Research Agent capable of understanding user queries, searching the web using multiple sources, scraping relevant content, analyzing information, and synthesizing comprehensive reports. It features a React frontend for user interaction and a Python FastAPI backend driving the agent's logic.

## Key Features

*   **Natural Language Query Understanding:** Leverages an LLM (e.g., GPT-4o-mini) to analyze user intent, break down complex queries, and determine search strategy.
*   **Hybrid Web Search:** Combines results from standard search (SerpApi), LLM-optimized search (Tavily), and news aggregation (NewsAPI).
*   **Semantic Re-ranking:** Improves search result relevance using sentence transformers.
*   **Respectful Web Scraping:** Extracts text, tables, and lists from websites while respecting `robots.txt` rules.
*   **Content Analysis:** Uses an LLM to assess relevance, summarize content, and extract specific data points.
*   **Information Synthesis:** Generates a coherent report answering the original query based on analyzed content.
*   **Streaming Interface:** Provides real-time updates on the research process via Server-Sent Events (SSE).
*   **React Frontend:** Modern UI with light/dark mode, search history, and clear presentation of results.

## Prerequisites

*   Python 3.8+
*   Node.js (v16 or later recommended) and npm (or yarn)
*   Access to an OpenAI API Key (or another supported LLM)
*   API Keys for search tools (Optional but recommended for full functionality):
    *   SerpApi
    *   Tavily
    *   NewsAPI

## Backend Setup

1.  **Navigate to Backend Directory:**
    ```bash
    cd backend
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:** Ensure you have the latest dependencies, including those for `robots.txt` handling and search APIs:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure Environment Variables:**
    *   Create a file named `.env` in the `backend` directory.
    *   Add your API keys to this file:
        ```dotenv
        OPENAI_API_KEY="your_openai_api_key"
        SERPAPI_API_KEY="your_serpapi_api_key"     # Optional
        TAVILY_API_KEY="your_tavily_api_key"       # Optional
        NEWS_API_KEY="your_newsapi_api_key"       # Optional
        ```
    *   Replace `"your_..._key"` with your actual API keys. The agent will still run without the optional keys but with reduced search capabilities.

## Frontend Setup

1.  **Navigate to Frontend Directory:**
    ```bash
    cd frontend
    ```

2.  **Install Dependencies:**
    ```bash
    npm install
    # OR if you use yarn:
    # yarn install
    ```
    *Note: This installs React, MUI, react-markdown, and other necessary frontend packages listed in `package.json`.*

## Running the Application

1.  **Start the Backend Server:**
    *   Make sure you are in the `backend` directory with your virtual environment activated.
    *   Run the FastAPI application using Uvicorn:
        ```bash
        uvicorn app.main:app --reload --port 8000
        ```
    *   The backend API will be running at `http://127.0.0.1:8000`. The `--reload` flag automatically restarts the server when code changes are detected.

2.  **Start the Frontend Development Server:**
    *   Open a *new* terminal window.
    *   Navigate to the `frontend` directory.
    *   Run the React development server:
        ```bash
        npm start
        # OR if you use yarn:
        # yarn start
        ```
    *   This will automatically open the application in your default web browser, usually at `http://localhost:3000`. The frontend is configured to proxy API requests to the backend running on port 8000.

3.  **Use the Agent:**
    *   Enter your research query in the search bar and click "Search".
    *   Observe the status updates below the search bar.
    *   View the synthesized report and source details once the process completes.

## Architecture & Design

For a detailed explanation of the agent's architecture, data flow, decision-making logic, tool integration, and error handling strategy, please refer to the [ARCHITECTURE.md](ARCHITECTURE.md) file.

## Technologies Used

*   **Backend:**
    *   Python
    *   FastAPI
    *   Uvicorn
    *   OpenAI API Client
    *   Requests
    *   BeautifulSoup4 + lxml
    *   Sentence-Transformers
    *   python-dotenv
    *   APIs: SerpApi, Tavily, NewsAPI
    *   RobotExclusionRulesParser
    *   (Core Dependencies: Pydantic, Starlette)
    *   (Search Libs: google-search-results, newsapi-python)
    *   (Browser Automation: Selenium, webdriver-manager) - *Note: Selenium usage might be limited/removed in final agent.py logic.*
*   **Frontend:**
    *   React
    *   Material UI (MUI)
    *   React Markdown
    *   CSS

</rewritten_file> 