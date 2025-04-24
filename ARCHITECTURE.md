# Web Research Agent: Architecture and Plan

This document outlines the step-by-step plan the `WebResearchAgent` follows to handle a user's research query from start to finish.

## Overall Goal

To take a natural language research query from a user, perform automated web searches and content analysis, and synthesize the findings from multiple sources into a coherent report that directly addresses the user's query, while respecting website access policies (`robots.txt`).

## Flowchart / Data Flow

```mermaid
graph TD
    A[User Query] --> B(1. Analyze Query);
    B --> C{Query Analysis Data\n(Intent, Sub-queries, Info Type, etc.)};
    C --> D(2. Run Web Search);
    D --> E{Search Results Collection\n(URLs, Snippets, Metadata)};
    E --> F(3. Aggregate URLs & Direct Content);
    F --> G{URLs to Scrape & Direct Content};
    G --> H{Check robots.txt};
    H -- Allowed --> I(4. Scrape/Crawl Page);
    H -- Disallowed --> J(Skip URL);
    I --> K{Scraped Content Collection\n(HTML Text, Tables, Lists, etc.)};
    J --> K; // Skipped URLs also tracked
    K --> L(5. Analyze Content);
    C --> L;  // Query Analysis Data also feeds into Content Analysis
    L --> M{Content Analysis Results\n(Relevance, Summaries, Extracted Data)};
    M --> N(6. Synthesize Results);
    A --> N;  // Original Query also feeds into Synthesis
    N --> O[Synthesized Report];

    subgraph Error Handling
        B -- Error --> Z(Return Error/Stop);
        D -- API/Key Error --> Z;
        H -- robots.txt Fetch Error --> J; // Assume disallowed on error
        I -- Scrape Error --> J; // Skip URL on scrape error
        L -- LLM/JSON Error --> Z(Partial Results Possible);
        N -- LLM Error --> Z(Return Error/Stop);
    end

    style Z fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#ffcc00,stroke:#333,stroke-width:1px
```
*Note: Mermaid syntax for rendering flowcharts. You can paste this into environments that support Mermaid (like some Markdown viewers, GitHub).* 

*Alternative Text Representation:*

```
[User Query]
     |
     v
[1. Analyze Query (LLM)] -> {Query Analysis Data}
     |
     v
[2. Run Web Search (APIs)] -> {Search Results Collection}
     |
     v
[3. Aggregate URLs & Direct Content] -> {URLs List & Direct Content}
     |
     v
Foreach URL in List:
  [ Check robots.txt (Cache/Fetch) ] -- Allowed --> [4. Scrape/Crawl Page] -> {Scraped Content}
     |                              `-- Disallowed -> [ Skip URL ]
     v
Accumulate -> {Scraped Content Collection}
     |
     +----------------------------- {Query Analysis Data}
     |
     v
[5. Analyze Content (LLM)] -> {Content Analysis Results}
     |
     +----------------------------- [Original Query]
     |
     v
[6. Synthesize Results (LLM)] -> [Final Report]

(Error handling occurs at each LLM/API/Scraping/robots.txt step)
```

## Tool Integration Summary

This section summarizes the key external tools and internal functions acting as tools, detailing their inputs, outputs, and how the agent utilizes them.

**1. Web Search - SerpApi (`perform_search`)**
   - **Input:** Sub-query (string), number of results (int), start index (int), API Key.
   - **Output:** Dictionary containing search results (including `organic_results` list with URLs, snippets, titles) or an error dictionary.
   - **Agent Usage:** Used for the initial web search for each sub-query. Provides foundational URLs and snippets. The agent checks for `organic_results` and potential errors.

**2. Web Search - Tavily (`perform_tavily_search`)**
   - **Input:** Sub-query (string), search depth (string, e.g., "basic"), max results (int), API Key.
   - **Output:** Dictionary containing optimized search results, potentially including a direct `answer` and a `results` list (URLs, content, titles).
   - **Agent Usage:** Used as the primary supplementary search tool for general queries (or fallback for news). The direct `answer` is extracted if present. `results` are combined with others. Provides diverse, LLM-focused search perspective.

**3. News Aggregator - NewsAPI (`get_news_articles`)**
   - **Input:** Sub-query (string), number of results/page size (int), page number (int), API Key.
   - **Output:** Dictionary containing news articles (list under `articles` key with URLs, descriptions, titles), total results count, and status.
   - **Agent Usage:** Used as the primary supplementary search tool for news-focused queries (or fallback for general). Provides recent, relevant news articles based on the query.

**4. Web Scraper/Crawler (`scrape_web_page`)**
   - **Input:** URL to scrape (string), User-Agent string.
   - **Output:** Dictionary containing scraped data (`url`, `text`, `tables` as list-of-lists, `lists` as list-of-lists) or `None` if scraping disallowed by robots.txt, fails (network error, non-HTML), or finds no meaningful content.
   - **Agent Usage:** Fetches content from URLs identified during search or crawling. Provides the raw text, table, and list data needed for content analysis. The agent handles the `None` output by skipping analysis for that URL.

**5. Content Analyzer (LLM via `analyze_content`)**
   - **Input:** Scraped content (dictionary with `text`, `tables`, `lists`, or just a string for snippets/answers), original user query (string), target data points (list, optional), news-focused flag (boolean).
   - **Output:** Dictionary containing analysis: `is_relevant` (boolean), `summary` (string or None), `extracted_data` (dict or None), and potential `error` message.
   - **Agent Usage:** Processes scraped content or direct answers/snippets using an LLM. Determines relevance to the original query, generates a concise summary focusing on relevant aspects, and extracts specific data points if requested. The output (`is_relevant`, `summary`, `extracted_data`) directly informs the final synthesis step.

**6. Semantic Re-ranker (SentenceTransformer via `_rerank_search_results`)**
   - **Input:** List of combined search result dictionaries, query analysis dictionary (for intent).
   - **Output:** Re-sorted list of search result dictionaries based on semantic similarity to the query intent.
   - **Agent Usage:** Improves the quality of search results passed to the scraping phase by prioritizing results semantically closer to the user's core intent, especially useful when combining results from different APIs (SerpApi, Tavily, NewsAPI).

## Agent Decision Making

Beyond the linear flow of the pipeline, the agent makes several key decisions based on the query analysis and intermediate results:

1.  **News vs. General Query Determination (`analyze_query` -> `run_web_search`):**
    *   **Decision:** Is the user query primarily about recent news or events?
    *   **Logic:** After the LLM analyzes the query and returns an `info_type`, the agent checks if `info_type` (converted to lowercase) contains keywords like "news", "latest", or "current events".
    *   **Outcome:** If keywords are present, the `is_news_focused` flag is set to `True`; otherwise, it's `False`.

2.  **Supplementary Search API Selection (`run_web_search`):**
    *   **Decision:** Which search API should be used to *supplement* the initial SerpApi results if more results are needed?
    *   **Logic:** Checks the `is_news_focused` flag.
    *   **Outcome:**
        *   If `is_news_focused` is `True`, the supplementary API chosen is NewsAPI (with Tavily as fallback).
        *   If `is_news_focused` is `False`, the supplementary API chosen is Tavily (with NewsAPI as fallback).
        *   The agent *always* tries SerpApi first for a small number of results if the key is available.

3.  **Search API Fallback (`run_web_search`):**
    *   **Decision:** Should the agent try a different search API *during the supplementary search phase*?
    *   **Logic:** Fallback occurs if:
        *   The required API key for the current supplementary API is missing.
        *   An API call to the current supplementary API fails (e.g., raises an exception).
        *   The first attempt with the current supplementary API returns no results.
    *   **Outcome:** The agent switches to the next API in the predefined fallback order (e.g., NewsAPI -> Tavily for news; Tavily -> NewsAPI for general). Only one fallback attempt is made per sub-query.

4.  **Search Pagination Termination (`run_web_search`):**
    *   **Decision:** Should the agent stop requesting more pages of results for the current sub-query and *supplementary* API?
    *   **Logic:** Pagination stops if:
        *   The total number of `results` collected (from SerpApi + supplementary) meets or exceeds the `total_results_target`.
        *   The maximum number of pages (`max_pages_per_query`) has been fetched *for the supplementary API*.
        *   An API call (for the supplementary API) returns no results on a page *after* the first page.
        *   Tavily is used (it's only called once).
    *   **Outcome:** The loop for the current sub-query/API terminates.

5.  **Robots.txt Compliance Check (`scrape_search_results`):**
    *   **Decision:** Is the agent allowed to scrape a specific `current_url`?
    *   **Logic:**
        *   Extracts the domain from `current_url`.
        *   Checks an internal `robots_cache` for the domain's rules.
        *   If not cached, calls `_get_robots_parser` helper:
            *   Constructs `robots.txt` URL.
            *   Fetches `robots.txt` using `requests` and the defined `USER_AGENT`.
            *   Handles responses (200=Parse, 404=Assume Allow, Other=Assume Disallow).
            *   Handles fetch/parse errors (Assume Disallow).
            *   Stores the resulting `RobotExclusionRulesParser` object (or a default permissive/restrictive one) in the cache.
        *   Calls the parser's `is_allowed(USER_AGENT, current_url)` method.
    *   **Outcome:** If `is_allowed` returns `False`, the URL is skipped, and scraping is not attempted. If `True` (or no rules found/error occurred resulting in an "allow" assumption by the parser), scraping proceeds. The same check is performed before adding new links found during crawling to the queue.

6.  **Crawling Strategy (`run_research_pipeline` -> `scrape_search_results`):**
    *   **Decision:** Should the agent attempt to crawl beyond the initial URLs found in search results?
    *   **Logic:** The `run_research_pipeline` method checks if the initial `query_analysis` identified specific `target_data_points`. *(Note: This specific decision logic was simplified in the current SSE streaming implementation in `main.py`, which uses fixed default limits unless explicitly passed different parameters).* 
    *   **Outcome:** If `target_data_points` were identified (in the original design), a shallow crawl (`crawl_depth=1`) might be enabled with potentially more pages allowed (`max_pages`). Otherwise, `crawl_depth` is 0 (scrape initial URLs only).

7.  **Content Analysis Prompt Selection (`analyze_content`):**
    *   **Decision:** Which LLM system prompt should be used for analyzing scraped content?
    *   **Logic:** Checks the `is_news_focused` flag passed down from the initial query analysis.
    *   **Outcome:** Selects either the `news_system_prompt_template` or the `base_system_prompt_template`.

8.  **Inclusion in Final Synthesis (`synthesize_results`):**
    *   **Decision:** Should the analysis results for a specific source (URL/snippet/answer) be included in the information sent to the final synthesis LLM?
    *   **Logic:** Checks the `is_relevant` flag from the `analyze_content` step for that source. Only includes sources marked as relevant that also have a non-empty `summary` OR non-empty `extracted_data`.
    *   **Outcome:** Only relevant, informative content is aggregated and passed to the synthesis prompt.

## Error Handling Strategy

The agent incorporates error handling at various stages to ensure robustness and provide informative feedback or graceful failure:

*   **LLM API Errors (`analyze_query`, `analyze_content`, `synthesize_results`):**
    *   **Problem:** Network issues, invalid API keys, rate limits, server errors from the LLM provider.
    *   **Handling:** `try...except openai.APIError` blocks wrap the LLM calls.
    *   **Outcome:** An error message is logged to the console. The specific method returns a default/error state (e.g., `analyze_query` returns an analysis dictionary indicating "Analysis failed" and includes the status code if available; `analyze_content` returns `is_relevant: False`; `synthesize_results` returns an error message string). The pipeline may continue with degraded information or terminate depending on the stage.

*   **LLM Response Format Errors (`analyze_query`, `analyze_content`):**
    *   **Problem:** The LLM returns a response that is not valid JSON or is missing required keys specified in the prompt.
    *   **Handling:** `try...except json.JSONDecodeError` and explicit key checking within the methods.
    *   **Outcome:** An error message is logged. The method returns a default/error state (similar to API errors). The pipeline may continue or terminate.

*   **Search API Errors/Failures (`run_web_search`):**
    *   **Problem:** Network issues, invalid API keys, rate limits, server errors from the Search API provider (NewsAPI, SerpApi, Tavily), or the API returning zero results on the first attempt.
    *   **Handling:** `try...except Exception` blocks wrap individual API tool calls. Logic explicitly checks for missing API keys before attempting calls. Checks for empty result lists (`articles`, `organic_results`) after a successful call.
    *   **Outcome:** An error message is logged. The agent attempts to use the next API in the predefined fallback order *for the supplementary search*. If all APIs fail for a sub-query, that sub-query yields no results. If *all* sub-queries yield no results, the pipeline may stop before scraping.

*   **Robots.txt Fetch/Parse Errors (`_get_robots_parser` called by `scrape_search_results`):**
    *   **Problem:** Network issues reaching `robots.txt`, timeouts, invalid file format.
    *   **Handling:** The `_get_robots_parser` helper catches `requests.exceptions.RequestException` and general `Exception`.
    *   **Outcome:** An error message is logged. A restrictive parser (`Disallow: /`) is cached for the domain (conservative approach), preventing scraping attempts on that domain for the current run. URLs from that domain are effectively skipped. If a 404 occurs, a permissive parser (`Allow: /`) is cached.

*   **Web Scraping Failures (`scrape_search_results` via `scrape_web_page`):**
    *   **Problem:** Websites are unreachable, return non-HTML content, block scraping attempts (e.g., via JavaScript challenges or HTTP errors), or timeout.
    *   **Handling:** The `scrape_web_page` tool function (assumed implementation) should catch common exceptions (e.g., `requests.exceptions.RequestException`) and return `None` if scraping fails.
    *   **Outcome:** The `scrape_search_results` method stores `None` as the content for the failed URL. An error entry (`"error": "Scraping failed..."`) is created during the Content Analysis stage for that URL. The pipeline continues, potentially relying on snippets, direct answers, or content from other successfully scraped pages.

*   **Conflicting Information (`synthesize_results`):**
    *   **Problem:** Different sources provide contradictory information (e.g., different prices, dates, conclusions).
    *   **Current Handling:** The system prompt for the `synthesize_results` LLM explicitly asks it to "acknowledge the discrepancy if significant".
    *   **Future Enhancement:** Implement pre-synthesis detection to identify conflicting values within `extracted_data` and add explicit flags to the synthesis prompt for the LLM to address.
    *   **Outcome (Current):** Relies on the synthesis LLM's ability to notice and report significant conflicts based on the prompt instruction.

*   **Pipeline Termination:**
    *   The pipeline is designed to stop early if critical initial steps fail without fallback options (e.g., Query Analysis completely fails to produce sub-queries) or if no searchable results are found at all. It attempts to continue past non-critical failures like individual page scraping errors.

## Pipeline Steps

The agent executes the research process through a defined pipeline (`run_research_pipeline` method):

**0. Initialization:**
   - The agent instance is created (`WebResearchAgent`).
   - Environment variables (API Keys) are loaded.
   - An LLM client (e.g., OpenAI) is initialized using the provided API key and configured model (e.g., `gpt-4o-mini`).
   - **New:** `robots_cache` dictionary initialized.

**1. Query Analysis (`analyze_query`):**
   - **Input:** Raw `user_query` (string).
   - **Process:**
     - Construct a prompt for the LLM, instructing it to act as a query analyzer.
     - The prompt asks the LLM to return a JSON object containing:
       - `intent`: Concise summary of the user's goal.
       - `sub_queries`: List of specific questions/keywords for web searches.
       - `info_type`: Primary type of information sought (e.g., "factual data", "latest news").
       - `target_data_points`: List of specific data items to look for (if any).
       - `search_strategy`: Brief plan for how to conduct the search.
     - Send the `user_query` and the system prompt to the configured LLM.
     - Attempt to parse the LLM response as JSON.
     - Perform basic validation (check for required keys, ensure `sub_queries` and `target_data_points` are lists).
     - Determine `is_news_focused` flag based on keywords ("news", "latest", "current events") in the `info_type`.
     - **Error Handling:** Catch LLM API errors, JSON parsing errors, and missing key errors. Return a default/error analysis structure if issues occur.
   - **Output:** `query_analysis` dictionary containing the parsed/validated LLM response plus the `is_news_focused` flag and `original_query`. If analysis failed, the pipeline may stop here.

**2. Web Search (`run_web_search`):**
   - **Input:** `query_analysis` dictionary, `total_results_target`, `max_pages_per_query`, `results_per_page`.
   - **Process:**
     - **Loop through each `sub_query`:**
       - Attempt initial search with SerpApi (if key available) for a small, fixed number of results (e.g., 3).
       - If `total_results_target` is not met:
         - Determine supplementary API based on `is_news_focused` (NewsAPI or Tavily).
         - Define fallback order (e.g., NewsAPI -> Tavily or Tavily -> NewsAPI).
         - Start pagination loop for the *supplementary* API:
           - Check for API key; if missing, attempt fallback.
           - **Attempt API Call:** Call supplementary API tool function (`get_news_articles` or `perform_tavily_search`).
           - **Error Handling:** Catch exceptions; if error or no results on first page, attempt fallback.
           - **Process Results:** Add results to `aggregated_results`. Stop if target met.
           - Stop pagination for the current API if max pages reached or no more results returned.
       - Combine initial SerpApi results with supplementary results.
       - Trim combined list to `total_results_target`.
       - Apply semantic re-ranking using SentenceTransformer model (if available).
       - Store the results: `api_used` (sorted list of unique APIs providing results), `results` (list of combined, trimmed, re-ranked result items), `metadata` (from first successful API call).
   - **Output:** `search_results_collection` dictionary mapping each `sub_query` to its results dictionary. If no results are found across all sub-queries and APIs, the pipeline may stop.

**3. Content Aggregation & Scraping (`scrape_search_results`):**
   - **Input:** `search_results_collection`, `max_pages`, `crawl_depth`, `stay_on_domain`.
   - **Process:**
     - Initialize `initial_urls` (set) and `direct_content` (dict).
     - **Aggregate URLs & Direct Content:** Loop through `search_results_collection` to extract URLs, snippets, and Tavily answers. Populate `initial_urls` and `direct_content`.
     - Initialize `scraped_content` dictionary by copying `direct_content`.
     - Initialize scraping `queue` (deque) with `initial_urls`.
     - Initialize `visited` set (`direct_content` keys + `initial_urls`).
     - Define `USER_AGENT` string.
     - **Loop while `queue` is not empty and `pages_attempted_scrape < max_pages`:**
       - Dequeue `(current_url, current_depth)`.
       - If `current_url` already processed (in `scraped_content`), continue.
       - **Check Robots.txt:**
         - Call `_get_robots_parser(current_url)` helper to get (cached or fetched) rules for the domain.
         - Use `parser.is_allowed(USER_AGENT, current_url)` to check permission.
         - If disallowed, log skip, store `None` for URL in `scraped_content`, and continue to next URL.
       - Increment `pages_attempted_scrape` (only counts pages *attempted* after passing robots check).
       - **Scrape Page:** Call `scrape_web_page(current_url, USER_AGENT)` tool function.
       - **Error Handling:** If scraping fails, `scrape_web_page` returns `None`. Store result (`dict` or `None`) in `scraped_content`.
       - **Crawling (if enabled & scrape successful):**
         - Parse scraped HTML.
         - Find links. Resolve/normalize URLs.
         - Filter links (HTTP/S, not visited, domain constraint).
         - **Check Robots.txt for New Links:** For each valid potential link, call `_get_robots_parser` and check `is_allowed` before adding to queue/visited. Skip if disallowed.
         - Add allowed links to `queue` and `visited`.
       - Apply politeness delay (`time.sleep(CRAWL_DELAY)`).
   - **Output:** `scraped_content` dictionary mapping URLs/keys to scraped data (`dict`), direct content (`str`), or `None` (if skipped/failed).

**4. Content Analysis (Loop over `analyze_content`):**
   - **Input:** `scraped_content` dictionary, `original_query`, `target_data_points`, `is_news_focused`.
   - **Process:**
     - Initialize `content_analysis_results` dictionary.
     - **Loop through each `item_key`, `text` in `scraped_content`:**
       - Determine `source_type` (Tavily answer, snippet, scraped page).
       - If `text` is empty/None (failed scrape), store an error entry.
       - If `source_type` is `tavily_answer`, create a result directly: `is_relevant: True`, `summary: text`, `extracted_data: None`.
       - **If text needs analysis (scraped page or snippet):**
         - Call `analyze_content(text, original_query, target_data_points, is_news_focused)`.
         - Inside `analyze_content`:
           - Select appropriate LLM system prompt (standard or news-focused).
           - Format prompt including text (truncated if necessary), original query, and data points request.
           - Send prompt to LLM, requesting JSON output (`is_relevant`, `summary`, `extracted_data`).
           - Parse LLM response.
           - **Error Handling:** Catch LLM API/JSON errors, return default analysis (`is_relevant: False`).
           - Validate response structure.
           - Return analysis dictionary.
       - Store the returned analysis dictionary (or error entry) in `content_analysis_results` using `item_key`.
   - **Output:** `content_analysis_results` dictionary mapping original `item_key`s to analysis dictionaries.

**5. Result Synthesis (`synthesize_results`):**
   - **Input:** `original_query`, `content_analysis_results`.
   - **Process:**
     - Initialize `relevant_content` list.
     - **Loop through `content_analysis_results`:**
       - If `analysis_data["is_relevant"]` is True and it contains a non-empty `summary` OR non-empty `extracted_data`:
         - Add a dictionary `{ "url": url, "summary": summary, "extracted_data": extracted_data }` to `relevant_content`.
     - If `relevant_content` is empty, return a "Synthesis failed: No relevant content..." message.
     - Construct `content_input_str` for the LLM by formatting each item in `relevant_content` (Source URL, Summary, Extracted Data).
     - Truncate `content_input_str` if it exceeds a maximum length.
     - Construct a system prompt instructing the LLM to act as a research report synthesizer, using *only* the provided content, answering the `original_query`, integrating summaries and data, acknowledging conflicts, and maintaining an objective tone.
     - Send the system prompt, `original_query`, and `content_input_str` to the LLM.
     - **Error Handling:** Catch LLM API errors. Return an error message if issues occur.
     - Get the synthesized report text from the LLM response.
   - **Output:** `synthesized_report` string.

**6. Final Output:**
   - Return a dictionary containing the results from all stages: `query_analysis`, `search_results`, `scraped_content`, `content_analysis`, `synthesized_report`. 