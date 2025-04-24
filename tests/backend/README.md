# WebResearchAgent Tests (`tests/backend/test_agent.py`)

This file contains unit tests for the `WebResearchAgent` class located in `backend/app/agent.py`. The tests use the `pytest` framework and the `pytest-mock` plugin to isolate the agent's logic from external dependencies like LLM APIs, Search APIs, and web scraping.

## Running the Tests

1.  Ensure you have a virtual environment set up for the project (`D:\WebResearchAgent`).
2.  Activate the virtual environment (e.g., `.\venv\Scripts\activate`).
3.  Make sure all development dependencies are installed (`pip install -r requirements.txt` and `pip install pytest pytest-mock`).
4.  From the project root directory (`D:\WebResearchAgent`), run:
    ```bash
    python -m pytest tests/backend
    ```

## Test Descriptions

The following tests are currently implemented:

**Agent Initialization:**

*   `test_agent_initialization(agent)`: Verifies that the `WebResearchAgent` can be instantiated without immediate errors and that its LLM client is correctly mocked.

**`analyze_query` Method:**

*   `test_analyze_query_success(agent, mocker)`: Tests the standard successful flow where the LLM call returns a valid analysis JSON.
*   `test_analyze_query_api_error(agent, mocker)`: Simulates an `openai.APIError` during the LLM call and verifies graceful handling and fallback.
*   `test_analyze_query_news_focus(agent, mocker)`: Tests if the agent correctly identifies a news-related query and sets the `is_news_focused` flag.
*   `test_analyze_query_llm_json_error(agent, mocker)`: Simulates the LLM returning invalid JSON and verifies error handling.
*   `test_analyze_query_llm_missing_keys(agent, mocker)`: Simulates the LLM returning JSON missing required keys and verifies error handling.

**`run_web_search` Method (Hybrid Approach):**

*   `test_run_web_search_hybrid_serpapi_only(agent, mocker)`: Tests the scenario where the initial SerpApi call meets the target number of results, so no supplementary search is needed.
*   `test_run_web_search_hybrid_serpapi_plus_news(agent, mocker)`: Tests a news-focused query where SerpApi provides initial results, and NewsAPI is successfully used to supplement them.
*   `test_run_web_search_hybrid_serpapi_plus_tavily(agent, mocker)`: Tests a general query where SerpApi provides initial results, and Tavily is successfully used to supplement them.
*   `test_run_web_search_hybrid_serpapi_fail_news_success(agent, mocker)`: Tests a scenario where the initial SerpApi call fails, and the agent successfully uses the appropriate supplementary API (NewsAPI in this test case).

**`scrape_search_results` Method:**

*   `test_scrape_search_results_simple(agent, mocker)`: Tests the aggregation of direct content (snippets, Tavily answers) and scraped content from mocked search results. Verifies that URLs are added correctly and the final collection contains the expected items based on mocked `scrape_web_page` calls.

**`analyze_content` Method:**

*   `test_analyze_content_relevant_no_data(agent, mocker)`: Tests content analysis for relevant text when no specific data points are requested.
*   `test_analyze_content_relevant_with_data(agent, mocker)`: Tests content analysis for relevant text when specific data points *are* requested, including handling missing data.
*   `test_analyze_content_irrelevant(agent, mocker)`: Tests content analysis when the provided text is irrelevant to the query.

**`synthesize_results` Method:**

*   `test_synthesize_results_success(agent, mocker)`: Tests the final report synthesis step. Provides mocked analysis results and mocks the synthesis LLM call. Verifies that the correct relevant information (summaries, extracted data) is passed to the LLM and the expected synthesized report is generated. 