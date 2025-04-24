# tests/backend/test_agent.py
import pytest
import os
import json # Added for mock response
from dotenv import load_dotenv
from unittest.mock import MagicMock # Alternative if mocker fixture isn't automatically available

# Import specific errors if needed for testing
from openai import APIError

# Add project root to allow importing backend module
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

from backend.app.agent import WebResearchAgent

# Load environment variables for tests if needed (e.g., for checking API key presence)
# Ensure you have a .env file at the project root or set environment variables
load_dotenv()

@pytest.fixture
def agent(mocker): # Add mocker fixture here
    """Provides a WebResearchAgent instance with a mocked LLM client."""
    # Mock the OpenAI client initialization to avoid real API key checks
    # and allow controlling its behavior in tests.
    mock_llm_client = MagicMock(spec=WebResearchAgent().llm_client) # Use MagicMock or mocker.Mock
    # We can pre-configure mock methods here if needed across many tests,
    # or patch them individually within each test.
    
    # Patch the OpenAI client *within the agent's initialization* if necessary,
    # but usually easier to patch the methods called by the agent.
    # For simplicity now, assume the agent initializes, and we patch its client's methods later.
    
    # Return an agent instance. We will mock its client's methods in tests.
    agent_instance = WebResearchAgent()
    # Replace the actual client with our mock *after* initialization
    agent_instance.llm_client = mock_llm_client 
    return agent_instance

def test_agent_initialization(agent):
    """Tests if the agent initializes without immediate errors."""
    assert agent is not None
    # Basic check: Does it have the core pipeline method?
    assert hasattr(agent, 'run_research_pipeline')
    # Check if the client is now a mock
    assert isinstance(agent.llm_client, MagicMock)
    print("Agent initialized successfully for testing with mocked LLM client.")

def test_analyze_query_success(agent, mocker): # Use mocker if not in fixture
    """Tests analyze_query with a mocked successful LLM response."""
    user_query = "What is the capital of France?"
    expected_analysis = {
        "intent": "Find the capital city of France.",
        "sub_queries": ["capital of France"],
        "info_type": "factual data",
        "target_data_points": ["capital city"],
        "search_strategy": "Direct factual lookup.",
        # These are added by analyze_query
        "original_query": user_query,
        "is_news_focused": False
    }

    # Mock the specific method call on the mocked client
    mock_create = agent.llm_client.chat.completions.create
    
    # Configure the return value structure expected from openai>=1.0
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message = MagicMock()
    mock_completion.choices[0].message.content = json.dumps(expected_analysis) # LLM returns JSON string
    mock_create.return_value = mock_completion

    # Run the method under test
    actual_analysis = agent.analyze_query(user_query)

    # Assertions
    mock_create.assert_called_once() # Verify the LLM call was made
    # Check that the result matches the structure returned by the LLM, plus added fields
    assert actual_analysis["intent"] == expected_analysis["intent"]
    assert actual_analysis["sub_queries"] == expected_analysis["sub_queries"]
    assert actual_analysis["info_type"] == expected_analysis["info_type"]
    assert actual_analysis["target_data_points"] == expected_analysis["target_data_points"]
    assert actual_analysis["search_strategy"] == expected_analysis["search_strategy"]
    assert actual_analysis["original_query"] == user_query
    assert not actual_analysis["is_news_focused"]
    print("test_analyze_query_success passed.")

def test_analyze_query_api_error(agent, mocker):
    """Tests analyze_query when the LLM call raises an APIError."""
    user_query = "Query that causes an error"
    error_status_code = 500
    error_message = "Simulated API Error"

    # Mock the specific method call to raise an error
    mock_create = agent.llm_client.chat.completions.create
    # Simulate an APIError (adjust details as needed)
    # Note: You might need to construct a mock response object for the error if the library expects it
    
    # 1. Create the mock response and set its status code
    mock_response = mocker.Mock()
    mock_response.status_code = error_status_code
    
    # 2. Instantiate APIError with arguments it accepts (e.g., message, body)
    #    The exact required arguments might vary slightly based on openai version specifics,
    #    but message and body are common.
    error_instance = APIError(
        message=error_message, 
        body=None, # Often needed, even if None
        request=mocker.Mock() # Sometimes a mock request is needed too
    )
    
    # 3. Attach the mock response to the error instance
    error_instance.response = mock_response
    
    # 4. Set the side_effect to raise the pre-configured error instance
    mock_create.side_effect = error_instance

    # Run the method under test
    actual_analysis = agent.analyze_query(user_query)

    # Assertions
    mock_create.assert_called_once()
    # Check if the result reflects the error handling logic in analyze_query
    assert "API Error" in actual_analysis["intent"]
    assert str(error_status_code) in actual_analysis["intent"]
    assert actual_analysis["sub_queries"] == [user_query] # Should default to original query
    assert actual_analysis["info_type"] == "unknown"
    assert actual_analysis["target_data_points"] == []
    assert "analysis failure" in actual_analysis["search_strategy"]
    assert not actual_analysis["is_news_focused"]
    assert actual_analysis["original_query"] == user_query
    print("test_analyze_query_api_error passed.")

def test_analyze_query_news_focus(agent, mocker):
    """Tests analyze_query correctly identifies a news-focused query."""
    user_query = "What are the latest developments in AI regulations?"
    expected_analysis = {
        "intent": "Find recent news about AI regulation changes.",
        "sub_queries": ["latest AI regulations news", "AI policy updates 2024"],
        "info_type": "latest news", # Key for triggering the flag
        "target_data_points": ["key regulations", "affected regions", "enforcement dates"],
        "search_strategy": "Search major news outlets and government press releases.",
        # is_news_focused is added by the agent based on info_type
    }

    # Mock the LLM call
    mock_create = agent.llm_client.chat.completions.create
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message = MagicMock()
    mock_completion.choices[0].message.content = json.dumps(expected_analysis)
    mock_create.return_value = mock_completion

    # Run the method
    actual_analysis = agent.analyze_query(user_query)

    # Assertions
    mock_create.assert_called_once()
    assert actual_analysis["info_type"] == expected_analysis["info_type"]
    assert actual_analysis["is_news_focused"] is True # Verify the flag
    assert actual_analysis["original_query"] == user_query
    print("test_analyze_query_news_focus passed.")

def test_analyze_query_llm_json_error(agent, mocker):
    """Tests analyze_query when the LLM returns invalid JSON."""
    user_query = "Query leading to JSON error"
    invalid_json_string = "This is not JSON{"

    # Mock the LLM call to return the invalid string
    mock_create = agent.llm_client.chat.completions.create
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message = MagicMock()
    mock_completion.choices[0].message.content = invalid_json_string
    mock_create.return_value = mock_completion

    # Run the method
    actual_analysis = agent.analyze_query(user_query)

    # Assertions
    mock_create.assert_called_once()
    # Check if the result reflects the JSON error handling
    assert "JSON parsing error" in actual_analysis["intent"]
    assert actual_analysis["sub_queries"] == [user_query] # Default fallback
    assert actual_analysis["info_type"] == "unknown"
    assert "analysis failure" in actual_analysis["search_strategy"]
    print("test_analyze_query_llm_json_error passed.")

def test_analyze_query_llm_missing_keys(agent, mocker):
    """Tests analyze_query when the LLM response is missing required keys."""
    user_query = "Query leading to missing keys"
    # Valid JSON, but missing 'info_type' and 'search_strategy'
    incomplete_analysis_dict = {
        "intent": "Partial intent",
        "sub_queries": ["partial query"],
        "target_data_points": []
        # Missing info_type, search_strategy
    }

    # Mock the LLM call
    mock_create = agent.llm_client.chat.completions.create
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message = MagicMock()
    mock_completion.choices[0].message.content = json.dumps(incomplete_analysis_dict)
    mock_create.return_value = mock_completion

    # Run the method
    actual_analysis = agent.analyze_query(user_query)

    # Assertions
    mock_create.assert_called_once()
    # Check if the result reflects the missing keys handling
    assert "format error" in actual_analysis["intent"] # Checks error message
    assert actual_analysis["sub_queries"] == [user_query] # Default fallback
    assert actual_analysis["info_type"] == "unknown" # Default fallback
    assert "analysis failure" in actual_analysis["search_strategy"] # Default fallback
    print("test_analyze_query_llm_missing_keys passed.")

# --- Tests for run_web_search --- 

# === OLD TESTS (COMMENTED OUT as logic changed) ===
# def test_run_web_search_news_success(agent, mocker):
#     # ... (old test code) ...
#     pass
# def test_run_web_search_general_success(agent, mocker):
#     # ... (old test code) ...
#     pass
# def test_run_web_search_news_fallback_to_tavily(agent, mocker):
#     # ... (old test code) ...
#     pass
# def test_run_web_search_news_fallback_to_serpapi(agent, mocker):
#     # ... (old test code) ...
#     pass
# def test_run_web_search_general_fallback_to_newsapi(agent, mocker):
#     # ... (old test code) ...
#     pass
# === END OLD TESTS ===

# --- NEW TESTS for HYBRID run_web_search --- 

INITIAL_SERPAPI_COUNT_FOR_TEST = 3 # Match the constant in agent code

def test_run_web_search_hybrid_serpapi_only(agent, mocker):
    """Tests hybrid search where initial SerpApi call meets target."""
    sub_query = "query needing 3 results"
    analysis_result = {"sub_queries": [sub_query], "is_news_focused": False}
    total_results_target = 3

    mock_serp_results = [
        {'title': 'S1', 'link': 's1', 'snippet': '...'},
        {'title': 'S2', 'link': 's2', 'snippet': '...'},
        {'title': 'S3', 'link': 's3', 'snippet': '...'}
    ]
    mock_serp_response = {'organic_results': mock_serp_results, 'search_metadata': {'status': 'Success'}}

    mock_perform_search = mocker.patch('backend.app.agent.perform_search', return_value=mock_serp_response)
    mock_get_news = mocker.patch('backend.app.agent.get_news_articles')
    mock_tavily_search = mocker.patch('backend.app.agent.perform_tavily_search')
    mocker.patch('backend.app.agent.SERPAPI_API_KEY', 'fake_serp_key')
    mocker.patch.object(agent, '_rerank_search_results', side_effect=lambda r, a: r) # Mock reranker to return original order for simplicity

    results = agent.run_web_search(analysis_result, total_results_target)

    mock_perform_search.assert_called_once_with(sub_query, num_results=INITIAL_SERPAPI_COUNT_FOR_TEST, start_index=0)
    mock_get_news.assert_not_called()
    mock_tavily_search.assert_not_called()
    assert results[sub_query]['api_used'] == ['serpapi']
    assert len(results[sub_query]['results']) == 3
    assert results[sub_query]['results'] == mock_serp_results
    print("test_run_web_search_hybrid_serpapi_only passed.")

def test_run_web_search_hybrid_serpapi_plus_news(agent, mocker):
    """Tests hybrid search: SerpApi + NewsAPI supplement for a news query."""
    sub_query = "latest news query"
    analysis_result = {"sub_queries": [sub_query], "is_news_focused": True}
    total_results_target = 5 # Need 5, Serp gives 2, News should provide 3

    mock_serp_results = [
        {'title': 'S1', 'link': 's1', 'snippet': 'Serp result 1'},
        {'title': 'S2', 'link': 's2', 'snippet': 'Serp result 2'}
    ]
    mock_serp_response = {'organic_results': mock_serp_results, 'search_metadata': {'status': 'Success'}}

    mock_news_articles = [
        {'title': 'N1', 'url': 'n1', 'description': 'News 1'},
        {'title': 'N2', 'url': 'n2', 'description': 'News 2'},
        {'title': 'N3', 'url': 'n3', 'description': 'News 3'}
    ]
    mock_news_response = {'articles': mock_news_articles, 'status': 'ok'}

    mock_perform_search = mocker.patch('backend.app.agent.perform_search', return_value=mock_serp_response)
    mock_get_news = mocker.patch('backend.app.agent.get_news_articles', return_value=mock_news_response)
    mock_tavily_search = mocker.patch('backend.app.agent.perform_tavily_search')
    mocker.patch('backend.app.agent.SERPAPI_API_KEY', 'fake_serp_key')
    mocker.patch('backend.app.agent.NEWS_API_KEY', 'fake_news_key')
    mocker.patch.object(agent, '_rerank_search_results', side_effect=lambda r, a: r) # Mock reranker

    results = agent.run_web_search(analysis_result, total_results_target, results_per_page=3) # results_per_page for supplementary

    mock_perform_search.assert_called_once_with(sub_query, num_results=INITIAL_SERPAPI_COUNT_FOR_TEST, start_index=0)
    # NewsAPI called once for the remaining 3 results (page_size=3)
    mock_get_news.assert_called_once_with(sub_query, num_results=3, page=1)
    mock_tavily_search.assert_not_called()
    assert results[sub_query]['api_used'] == ['newsapi', 'serpapi'] # Alphabetical
    assert len(results[sub_query]['results']) == 5
    # Check if results are combined (order depends on reranker mock - here it's sequential)
    assert results[sub_query]['results'] == mock_serp_results + mock_news_articles
    print("test_run_web_search_hybrid_serpapi_plus_news passed.")

def test_run_web_search_hybrid_serpapi_plus_tavily(agent, mocker):
    """Tests hybrid search: SerpApi + Tavily supplement for a general query."""
    sub_query = "general query supplement"
    analysis_result = {"sub_queries": [sub_query], "is_news_focused": False}
    total_results_target = 4 # Need 4, Serp gives 1, Tavily should provide 3

    mock_serp_results = [
        {'title': 'S1', 'link': 's1', 'snippet': 'Serp result 1'}
    ]
    mock_serp_response = {'organic_results': mock_serp_results, 'search_metadata': {'status': 'Success'}}

    mock_tavily_results = [
        {'title': 'T1', 'url': 't1', 'content': 'Tavily 1'},
        {'title': 'T2', 'url': 't2', 'content': 'Tavily 2'},
        {'title': 'T3', 'url': 't3', 'content': 'Tavily 3'}
    ]
    mock_tavily_response = {'results': mock_tavily_results, 'answer': 'Tavily answer.'}

    mock_perform_search = mocker.patch('backend.app.agent.perform_search', return_value=mock_serp_response)
    mock_get_news = mocker.patch('backend.app.agent.get_news_articles')
    mock_tavily_search = mocker.patch('backend.app.agent.perform_tavily_search', return_value=mock_tavily_response)
    mocker.patch('backend.app.agent.SERPAPI_API_KEY', 'fake_serp_key')
    mocker.patch('backend.app.agent.TAVILY_API_KEY', 'fake_tavily_key')
    mocker.patch.object(agent, '_rerank_search_results', side_effect=lambda r, a: r) # Mock reranker

    results = agent.run_web_search(analysis_result, total_results_target)

    mock_perform_search.assert_called_once_with(sub_query, num_results=INITIAL_SERPAPI_COUNT_FOR_TEST, start_index=0)
    mock_tavily_search.assert_called_once_with(sub_query, max_results=3) # remaining_target = 4 - 1 = 3
    mock_get_news.assert_not_called()
    assert results[sub_query]['api_used'] == ['serpapi', 'tavily'] # Alphabetical
    assert len(results[sub_query]['results']) == 4
    assert results[sub_query]['results'] == mock_serp_results + mock_tavily_results
    print("test_run_web_search_hybrid_serpapi_plus_tavily passed.")

def test_run_web_search_hybrid_serpapi_fail_news_success(agent, mocker):
    """Tests hybrid search: SerpApi fails, NewsAPI provides results for news query."""
    sub_query = "news query serp fail"
    analysis_result = {"sub_queries": [sub_query], "is_news_focused": True}
    total_results_target = 2

    # Mock NewsAPI results 
    mock_articles = [
        {'title': 'N1', 'url': 'n1', 'description': 'News 1'},
        {'title': 'N2', 'url': 'n2', 'description': 'News 2'}
    ]
    mock_news_response = {'articles': mock_articles, 'status': 'ok'}

    mock_perform_search = mocker.patch('backend.app.agent.perform_search', return_value=None) # SerpApi fails
    mock_get_news = mocker.patch('backend.app.agent.get_news_articles', return_value=mock_news_response)
    mock_tavily_search = mocker.patch('backend.app.agent.perform_tavily_search')
    mocker.patch('backend.app.agent.SERPAPI_API_KEY', 'fake_serp_key')
    mocker.patch('backend.app.agent.NEWS_API_KEY', 'fake_news_key')
    mocker.patch.object(agent, '_rerank_search_results', side_effect=lambda r, a: r) # Mock reranker

    results = agent.run_web_search(analysis_result, total_results_target, results_per_page=2)

    mock_perform_search.assert_called_once()
    mock_get_news.assert_called_once_with(sub_query, num_results=2, page=1) # Needs target=2
    mock_tavily_search.assert_not_called()
    assert results[sub_query]['api_used'] == ['newsapi'] # Only NewsAPI succeeded
    assert len(results[sub_query]['results']) == 2
    assert results[sub_query]['results'] == mock_articles
    print("test_run_web_search_hybrid_serpapi_fail_news_success passed.")

# --- Tests for scrape_search_results --- 

def test_scrape_search_results_simple(agent, mocker):
    """Tests basic scraping of URLs and handling of snippets/answers."""
    # 1. Prepare Inputs
    query1 = "query one"
    query2 = "query two (tavily)"
    url1 = "http://example.com/page1"
    url2 = "http://example.com/page2"
    snippet1_key = f"serpapi_snippet::{url1}"
    tavily_answer_key = f"tavily_answer::{query2}"

    search_results_collection = {
        query1: {
            'api_used': 'serpapi',
            'results': [
                {'title': 'Page 1', 'link': url1, 'snippet': 'Snippet for page 1...'},
                {'title': 'Page 2', 'link': url2, 'snippet': None} # No snippet
            ],
            'metadata': {}
        },
        query2: {
            'api_used': 'tavily',
            'results': [], # No separate results, just the answer
            'metadata': {'answer': 'Direct Tavily answer.'}
        }
    }

    # Define expected structured data from scraping
    mock_structured_data_url1 = {
        'text': "<html><body>Scraped content for page 1</body></html>",
        'tables': [[['Header'], ['Cell']]], # Example table
        'lists': []
    }
    mock_structured_data_url2 = {
        'text': "<html><body>Content for page 2 (no snippet)</body></html>",
        'tables': [],
        'lists': [['Item A', 'Item B']] # Example list
    }

    # 2. Mock Tool Functions
    # Mock scrape_web_page to return the structured dict
    def mock_scrape_logic(url):
        if url == url1:
            return mock_structured_data_url1
        elif url == url2:
            return mock_structured_data_url2
        return None # Default
    mock_scrape = mocker.patch('backend.app.agent.scrape_web_page', side_effect=mock_scrape_logic)
    mocker.patch('time.sleep')

    # 3. Run Method Under Test
    scraped_content = agent.scrape_search_results(
        search_results_collection,
        max_pages=5,
        crawl_depth=0
    )

    # 4. Assertions
    # Check that scrape_web_page was called for the URLs needing scraping
    # It shouldn't be called for URLs associated with direct answers (like Tavily answer sources, though none here)
    # It should be called for url1 and url2
    assert mock_scrape.call_count == 2
    mock_scrape.assert_any_call(url1)
    mock_scrape.assert_any_call(url2)

    # Check the output dictionary structure
    assert len(scraped_content) == 4 # tavily_answer + snippet1 + scraped url1 + scraped url2
    assert scraped_content[tavily_answer_key] == 'Direct Tavily answer.'
    
    # Update assertion to match generic snippet key format
    generic_snippet1_key = f"snippet::{url1}"
    assert generic_snippet1_key in scraped_content # Check key exists
    assert scraped_content[generic_snippet1_key] == 'Snippet: Snippet for page 1...'

    # Check scraped content for the URLs
    assert scraped_content[url1] == mock_structured_data_url1
    assert scraped_content[url2] == mock_structured_data_url2
    print("test_scrape_search_results_simple passed.")

# --- Tests for analyze_content ---

def test_analyze_content_relevant_no_data(agent, mocker):
    """Tests analyze_content identifies relevant text, no specific data requested."""
    # 1. Inputs
    # Provide structured data as input
    scraped_data_input = {
        'text': "This text discusses the main topic of the query in detail.",
        'tables': [],
        'lists': []
    }
    original_query = "Tell me about the main topic."
    target_data_points = []
    is_news_focused = False

    # 2. Mock LLM Response
    mock_llm_response_dict = {
        "is_relevant": True,
        "summary": "The text is relevant and covers the main topic.",
        "extracted_data": {}
    }
    mock_create = agent.llm_client.chat.completions.create
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message = MagicMock()
    mock_completion.choices[0].message.content = json.dumps(mock_llm_response_dict)
    mock_create.return_value = mock_completion

    # 3. Run Method (pass structured data)
    analysis = agent.analyze_content(scraped_data_input, original_query, target_data_points, is_news_focused)

    # 4. Assertions
    mock_create.assert_called_once()
    assert analysis["is_relevant"] is True
    assert analysis["summary"] == mock_llm_response_dict["summary"]
    assert analysis["extracted_data"] == {}
    print("test_analyze_content_relevant_no_data passed.")

def test_analyze_content_relevant_with_data(agent, mocker):
    """Tests analyze_content finds relevant text/structured data and extracts requested data."""
    # 1. Inputs
    # Add a mock table to the structured data
    mock_table = [["Metric", "Value"], ["Price", "$50"], ["Rating", "4.5 stars"]]
    scraped_data_input = {
        'text': "Product details are below.",
        'tables': [mock_table],
        'lists': []
    }
    original_query = "Find product price and rating."
    target_data_points = ["price", "rating", "availability"]
    is_news_focused = False

    # 2. Mock LLM Response
    mock_llm_response_dict = {
        "is_relevant": True,
        "summary": "The text and table provide price and rating.",
        "extracted_data": {
            "price": "$50",
            "rating": "4.5 stars",
            "availability": None
        }
    }
    mock_create = agent.llm_client.chat.completions.create
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message = MagicMock()
    mock_completion.choices[0].message.content = json.dumps(mock_llm_response_dict)
    mock_create.return_value = mock_completion

    # 3. Run Method
    analysis = agent.analyze_content(scraped_data_input, original_query, target_data_points, is_news_focused)

    # 4. Assertions
    mock_create.assert_called_once()
    assert analysis["is_relevant"] is True
    assert analysis["summary"] == mock_llm_response_dict["summary"]
    assert analysis["extracted_data"] is not None
    assert analysis["extracted_data"]["price"] == "$50"
    assert analysis["extracted_data"]["rating"] == "4.5 stars"
    assert analysis["extracted_data"]["availability"] is None
    
    # Verify formatted table was sent in the prompt
    call_args, call_kwargs = mock_create.call_args
    messages = call_kwargs.get('messages', [])
    user_prompt_message = next((m['content'] for m in messages if m['role'] == 'user'), None)
    assert user_prompt_message is not None
    assert "--- PARSED TABLES ---" in user_prompt_message
    assert "| Metric | Value |" in user_prompt_message
    assert "| --- | --- |" in user_prompt_message
    assert "| Price | $50 |" in user_prompt_message
    assert "| Rating | 4.5 stars |" in user_prompt_message
    
    print("test_analyze_content_relevant_with_data passed.")

def test_analyze_content_irrelevant(agent, mocker):
    """Tests analyze_content correctly identifies irrelevant text."""
    # 1. Inputs
    # Use structured data input
    scraped_data_input = {
        'text': "This text talks about unrelated matters.",
        'tables': [],
        'lists': []
    }
    original_query = "Tell me about the main topic."
    target_data_points = ["price"]
    is_news_focused = False

    # 2. Mock LLM Response
    mock_llm_response_dict = {
        "is_relevant": False,
        "summary": None,
        "extracted_data": {}
    }
    mock_create = agent.llm_client.chat.completions.create
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message = MagicMock()
    mock_completion.choices[0].message.content = json.dumps(mock_llm_response_dict)
    mock_create.return_value = mock_completion

    # 3. Run Method
    analysis = agent.analyze_content(scraped_data_input, original_query, target_data_points, is_news_focused)

    # 4. Assertions
    mock_create.assert_called_once()
    assert analysis["is_relevant"] is False
    assert analysis["summary"] is None
    assert analysis["extracted_data"] is None
    print("test_analyze_content_irrelevant passed.")

# --- Tests for synthesize_results ---

def test_synthesize_results_success(agent, mocker):
    """Tests synthesize_results combines summaries and data into a report."""
    # 1. Inputs
    original_query = "What is the price and summary for Product X?"
    url1 = "http://source1.com/productX"
    url2 = "http://source2.com/detailsX"
    content_analysis_results = {
        url1: {
            "is_relevant": True,
            "summary": "Source 1 says Product X is great.",
            "extracted_data": {"rating": "5 stars"},
            "source_type": "scraped_page"
        },
        url2: {
            "is_relevant": True,
            "summary": "Source 2 mentions the price.",
            "extracted_data": {"price": "$99"},
            "source_type": "scraped_page"
        },
        "http://irrelevant.com": {
            "is_relevant": False,
            "summary": None,
            "extracted_data": None
        }
    }

    expected_report_text = "Product X is great, according to Source 1 which gave it 5 stars. Source 2 mentions the price is $99."

    # 2. Mock LLM Call
    mock_create = agent.llm_client.chat.completions.create
    mock_completion = MagicMock()
    mock_completion.choices = [MagicMock()]
    mock_completion.choices[0].message = MagicMock()
    mock_completion.choices[0].message.content = expected_report_text
    mock_create.return_value = mock_completion

    # 3. Run Method
    final_report = agent.synthesize_results(original_query, content_analysis_results)

    # 4. Assertions
    mock_create.assert_called_once()
    
    # Verify the prompt content sent to the LLM
    call_args, call_kwargs = mock_create.call_args
    messages = call_kwargs.get('messages', [])
    user_prompt_message = next((m['content'] for m in messages if m['role'] == 'user'), None)
    assert user_prompt_message is not None
    # Check if key information from relevant sources is in the prompt
    assert url1 in user_prompt_message
    assert "Source 1 says Product X is great." in user_prompt_message
    assert '"rating": "5 stars"' in user_prompt_message # Check extracted data format
    assert url2 in user_prompt_message
    assert "Source 2 mentions the price." in user_prompt_message
    assert '"price": "$99"' in user_prompt_message
    assert "http://irrelevant.com" not in user_prompt_message # Irrelevant source excluded
    
    # Verify the returned report
    assert expected_report_text in final_report # Check if LLM response is there
    
    print("test_synthesize_results_success passed.")


# Example of a test structure (needs mocking) - KEEPING THIS FOR REFERENCE
# @pytest.mark.asyncio
# async def test_analyze_query_simple(agent, mocker):
#     """Tests the query analysis functionality with mocking."""
#     mock_response = {
#         "intent": "Test intent",
#         "sub_queries": ["test query"],
#         "info_type": "factual data",
#         "target_data_points": [],
#         "search_strategy": "Test strategy",
#     }
#     # Mock the OpenAI client's chat.completions.create method
#     mock_llm_call = mocker.patch.object(agent.llm_client.chat.completions, 'create', autospec=True)
#     # Configure the mock return value
#     mock_llm_call.return_value.choices[0].message.content = json.dumps(mock_response)

#     user_query = "This is a test query"
#     analysis = agent.analyze_query(user_query)
    
#     assert analysis["intent"] == mock_response["intent"]
#     assert analysis["sub_queries"] == mock_response["sub_queries"]
#     mock_llm_call.assert_called_once() # Ensure the mock was called
#     print("Mocked query analysis test structure executed.")


# Add more tests here covering:
# - Different query types (news, specific data)
# - Search result handling (different APIs used)
# - Scraping simulation/mocking
# - Content analysis logic (relevance, data extraction)
# - Synthesis process
# - Error handling (API errors, scraping failures) 