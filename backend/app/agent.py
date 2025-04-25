import os
from dotenv import load_dotenv
import json
from openai import OpenAI, APIError
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import time
from collections import deque
from tavily import TavilyClient # Added
from sentence_transformers import SentenceTransformer, util # Added

# --- NEW IMPORTS for robots.txt --- 
import requests
from robotexclusionrulesparser import RobotExclusionRulesParser
# --- END NEW IMPORTS ---

# Import the tools using relative import
from .tools import perform_search, scrape_web_page, get_news_articles, perform_tavily_search
# Import API Key constants from tools
from .tools import TAVILY_API_KEY, NEWS_API_KEY, SERPAPI_API_KEY

# Load environment variables (like API keys)
load_dotenv()

# Constants
CRAWL_DELAY = 1 # seconds between requests during crawl
# --- NEW CONSTANT --- 
USER_AGENT = "WebResearchAgent/1.0 (+https://github.com/your-repo-if-public)" # Define your agent's user agent
# --- END NEW CONSTANT --- 

class WebResearchAgent:
    def __init__(self, llm_client=None, model_name="gpt-4o-mini", reranker_model_name='all-MiniLM-L6-v2'):
        """
        Initializes the agent.
        Args:
            llm_client: An instance of the LLM client. If None, initializes OpenAI client.
            model_name: The OpenAI model to use for generation/analysis.
            reranker_model_name: The sentence-transformer model for re-ranking.
        """
        self.model_name = model_name
        # Initialize OpenAI client
        if llm_client:
            self.llm_client = llm_client
        else:
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not found in .env file or environment variables.")
                self.llm_client = OpenAI(api_key=api_key)
                print(f"OpenAI client initialized successfully using model: {self.model_name}")
            except Exception as e:
                print(f"Error initializing OpenAI client: {e}")
                self.llm_client = None

        # Check for SerpAPI key for search functionality
        if not os.getenv("SERPAPI_API_KEY"):
             print("Warning: SERPAPI_API_KEY not found. Real web search will not work.")

        if not self.llm_client:
            print("Warning: LLM client could not be initialized. Query analysis will be simulated.")

        # --- Initialize Re-ranker Model ---
        self.reranker_model_name = reranker_model_name
        self.reranker_model = None
        # --- Temporarily Disable Model Loading to Save Memory ---
        # try:
        #     # Load the model upon initialization
        #     print(f"Loading re-ranking model: {self.reranker_model_name}...")
        #     self.reranker_model = SentenceTransformer(self.reranker_model_name)
        #     print("Re-ranking model loaded successfully.")
        # except Exception as e:
        #     print(f"Error loading sentence-transformer model '{self.reranker_model_name}': {e}")
        #     print("Warning: Semantic re-ranking will be disabled.")
        #     self.reranker_model = None # Ensure it's None if loading failed
        print(f"Skipping loading of re-ranking model ({self.reranker_model_name}) to save memory.")
        # --- End Temporary Disable ---

        # --- NEW: Initialize robots.txt cache --- 
        self.robots_cache = {}
        # --- END NEW --- 

    def _initialize_llm_client(self):
        """Helper to initialize the LLM client."""
        if self.llm_client: # If client was passed in, use it
            print(f"Using provided LLM client: {type(self.llm_client).__name__}")
            return
        
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in .env file or environment variables.")
            self.llm_client = OpenAI(api_key=api_key)
            print(f"OpenAI client initialized successfully using model: {self.model_name}")
        except Exception as e:
            print(f"Error initializing OpenAI client: {e}")
            self.llm_client = None
        
        if not SERPAPI_API_KEY:
             print("Warning: SERPAPI_API_KEY not found. Real web search will be limited.")
             
        if not self.llm_client:
            print("Warning: LLM client could not be initialized. Query analysis/synthesis will be simulated or fail.")

    def _load_reranker_model(self):
        """Helper to load the re-ranking model."""
        try:
            print(f"Loading re-ranking model: {self.reranker_model_name}...")
            self.reranker_model = SentenceTransformer(self.reranker_model_name)
            print("Re-ranking model loaded successfully.")
        except Exception as e:
            print(f"Error loading sentence-transformer model '{self.reranker_model_name}': {e}")
            print("Warning: Semantic re-ranking will be disabled.")
            self.reranker_model = None

    # --- NEW: Helper for robots.txt --- 
    def _get_robots_parser(self, url: str) -> RobotExclusionRulesParser | None:
        """Fetches, parses, and caches robots.txt for a given URL's domain."""
        parsed_url = urlparse(url)
        domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = urljoin(domain, "/robots.txt")

        if domain in self.robots_cache:
            return self.robots_cache[domain]

        print(f"  -> Fetching robots.txt for: {domain}")
        parser = RobotExclusionRulesParser()
        try:
            headers = {'User-Agent': USER_AGENT}
            response = requests.get(robots_url, headers=headers, timeout=5)
            if response.status_code == 200:
                parser.parse(response.text)
                print(f"  -> Parsed robots.txt for {domain}")
            elif response.status_code == 404:
                 print(f"  -> No robots.txt found for {domain} (404), assuming allowed.")
                 # Store a parser that allows everything if 404
                 parser.parse("User-agent: *\nAllow: /") 
            else:
                 print(f"  -> robots.txt fetch failed for {domain}, status: {response.status_code}, assuming disallowed.")
                 # Store a parser that disallows everything on failure (conservative)
                 parser.parse("User-agent: *\nDisallow: /")
        except requests.exceptions.RequestException as e:
            print(f"  -> Error fetching robots.txt for {domain}: {e}, assuming disallowed.")
            # Store a parser that disallows everything on error
            parser.parse("User-agent: *\nDisallow: /")
        except Exception as e:
             print(f"  -> Unexpected error processing robots.txt for {domain}: {e}, assuming disallowed.")
             parser.parse("User-agent: *\nDisallow: /")
        
        self.robots_cache[domain] = parser
        return parser
    # --- END NEW HELPER --- 

    def analyze_query(self, user_query: str) -> dict:
        """
        Analyzes the user's research query using the configured LLM
        to understand intent, components, information type, and formulate a search strategy.

        Args:
            user_query: The research query provided by the user.

        Returns:
            A dictionary containing the analysis results.
            Returns a dictionary with placeholder/error values if LLM call fails.
        """
        print(f"Analyzing query using {self.model_name}: {user_query}")

        # --- Phase 1: Query Analysis ---

        # 1. Design the prompt for the LLM
        system_prompt = """You are an expert query analyzer for a web research agent.
        Your task is to analyze the user's query and extract key information to guide the research process.
        Return the analysis STRICTLY as a JSON object with the following keys:
        - "intent": A concise summary of the user's primary research goal.
        - "sub_queries": A list of specific questions or keywords for web searches. Break down complex queries.
        - "info_type": The primary type of information sought (e.g., "factual data", "latest news", "opinions", "historical context", "comparison", "general summary").
        - "target_data_points": A list of specific data items the user is looking for (e.g., ["price", "release date", "CEO name"]). If the query is general, return an empty list [].
        - "search_strategy": A brief plan for how the agent should approach the research (e.g., search sources, comparison points).
        Ensure the output is only the JSON object, without any introductory text or explanation."""

        user_prompt = f'Analyze the following user query: "{user_query}"'

        analysis_result = {
            "original_query": user_query,
            "intent": "Analysis failed",
            "sub_queries": [user_query], # Default to user query on failure
            "info_type": "unknown",
            "target_data_points": [], # NEW: Add target data points field
            "search_strategy": "Basic web search due to analysis failure.",
            "is_news_focused": False # NEW: Add flag
        }

        # 2. Call the LLM
        if self.llm_client and isinstance(self.llm_client, OpenAI):
            try:
                response = self.llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    response_format={"type": "json_object"}
                )
                response_content = response.choices[0].message.content
                if response_content:
                    parsed_analysis = json.loads(response_content)
                    required_keys = {"intent", "sub_queries", "info_type", "search_strategy", "target_data_points"}
                    if required_keys.issubset(parsed_analysis.keys()):
                         analysis_result.update(parsed_analysis)
                         # Ensure sub_queries is always a list
                         if not isinstance(analysis_result.get("sub_queries"), list):
                             print("Warning: LLM returned non-list for sub_queries. Wrapping in list.")
                             analysis_result["sub_queries"] = [analysis_result.get("sub_queries", user_query)]
                         elif not analysis_result.get("sub_queries"): # Handle empty list
                             print("Warning: LLM returned empty list for sub_queries. Using original query.")
                             analysis_result["sub_queries"] = [user_query]

                         # NEW: Ensure target_data_points is a list
                         if not isinstance(analysis_result.get("target_data_points"), list):
                             print("Warning: LLM returned non-list for target_data_points. Defaulting to empty list.")
                             analysis_result["target_data_points"] = []

                         # NEW: Set news flag based on info_type
                         info_type_lower = str(analysis_result.get("info_type", "")).lower()
                         if "news" in info_type_lower or "latest" in info_type_lower or "current events" in info_type_lower:
                             analysis_result["is_news_focused"] = True
                             print("Query identified as potentially news-focused.")
                         else:
                              analysis_result["is_news_focused"] = False # Ensure it's set back if overwritten by update()

                         print("LLM Query Analysis Successful.")
                    else:
                         print("Error: LLM response missing required keys.")
                         analysis_result["intent"] = "LLM response format error"
                         analysis_result["sub_queries"] = [user_query]
                else:
                    print("Error: LLM returned empty content.")
                    analysis_result["intent"] = "LLM returned empty content"
                    analysis_result["sub_queries"] = [user_query]
            except APIError as e:
                print(f"OpenAI API Error during query analysis: {e}")
                # Access status_code via the response attribute
                status_code = getattr(e.response, 'status_code', 'N/A') # Safely get status code
                analysis_result["intent"] = f"OpenAI API Error: {status_code}"
                analysis_result["sub_queries"] = [user_query]
            except json.JSONDecodeError as e:
                 print(f"Error parsing LLM JSON response: {e}")
                 analysis_result["intent"] = "LLM response JSON parsing error"
                 analysis_result["sub_queries"] = [user_query]
            except Exception as e:
                print(f"Unexpected error during LLM call for query analysis: {e}")
                analysis_result["intent"] = f"Unexpected analysis error: {type(e).__name__}"
                analysis_result["sub_queries"] = [user_query]
        else:
            print("LLM client not available or not OpenAI. Using placeholder analysis.")
            # Simulation logic (remains as fallback)
            if "quantum computing" in user_query.lower():
                 analysis_result.update({
                    "intent": "Find the latest developments and breakthroughs in quantum computing.",
                    "sub_queries": ["latest advancements quantum computing", "quantum computing breakthroughs 2024", "applications of quantum computing"],
                    "info_type": "latest news and developments",
                    "target_data_points": ["key breakthroughs", "research institutions involved", "potential applications"],
                    "search_strategy": "Search academic journals, tech news sites, and research institution websites for recent publications and announcements.",
                    "is_news_focused": True # Update simulation
                })
            elif "electric cars vs gasoline cars" in user_query.lower():
                 analysis_result.update({
                    "intent": "Compare the environmental impact of electric cars versus gasoline cars.",
                    "sub_queries": ["environmental impact electric cars", "environmental impact gasoline cars", "lifecycle emissions electric vs gasoline cars", "battery production environmental cost"],
                    "info_type": "comparison and factual data",
                    "target_data_points": ["CO2 emissions per mile (electric)", "CO2 emissions per mile (gasoline)", "battery production impact", "electricity source impact"],
                    "search_strategy": "Search for scientific studies, government reports (e.g., EPA), and reputable environmental organizations for data on lifecycle emissions, resource extraction, and energy sources.",
                    "is_news_focused": False # Update simulation
                })
            else:
                 analysis_result.update({
                     "intent": "Analyze the user query (simulated).",
                     "sub_queries": [user_query],
                     "info_type": "unknown (simulated)",
                     "target_data_points": [],
                     "search_strategy": "Perform basic web search (simulated).",
                     "is_news_focused": False # Update simulation
                 })

        # Ensure sub_queries is a list, even if analysis failed or was simulated
        if not isinstance(analysis_result.get("sub_queries"), list) or not analysis_result.get("sub_queries"):
            analysis_result["sub_queries"] = [user_query]

        print("Query Analysis Complete:")
        print(json.dumps(analysis_result, indent=2))
        return analysis_result

    # --- REVISED: Re-ranking Method (Semantic Similarity for Hybrid Results) ---
    def _rerank_search_results(self, results: list, analysis: dict) -> list:
        """Re-ranks a list of potentially mixed search results based on semantic similarity."""
        if not results or not analysis or not self.reranker_model:
            if not self.reranker_model and results:
                 print("  -> Skipping re-ranking: Model not loaded.")
            return results

        query_intent = analysis.get("intent", analysis.get("original_query", ""))
        if not query_intent:
            return results

        # Prepare result texts, handling key variations
        result_texts = []
        valid_results_with_text = []
        original_indices = [] # Keep track of original index for stable sort later if needed
        
        print(f"  -> Preparing {len(results)} results for semantic re-ranking...")
        for idx, result in enumerate(results):
            if not isinstance(result, dict):
                 print(f"  -> Skipping non-dict item in results list: {result}")
                 continue # Skip if item is not a dictionary
                 
            # --- Extract common fields with fallbacks --- 
            title = result.get('title', '')
            # Link is primarily for context, not ranking text
            # link = result.get('link') or result.get('url') 
            snippet = result.get('snippet') or result.get('content') or result.get('description', '')
            # --- End Extraction --- 
            
            combined_text = f"{str(title or '')} {' - ' if title and snippet else ''} {str(snippet or '')}".strip()
            
            if combined_text:
                 result_texts.append(combined_text)
                 valid_results_with_text.append(result)
                 original_indices.append(idx) # Store original index
            else:
                 print(f"  -> Skipping result with no text content: {result.get('link') or result.get('url')}")

        if not valid_results_with_text:
            return results

        try:
            # --- Calculate Embeddings and Similarity ---
            print(f"  -> Calculating semantic similarity for {len(valid_results_with_text)} results...")
            query_embedding = self.reranker_model.encode(query_intent, convert_to_tensor=True)
            result_embeddings = self.reranker_model.encode(result_texts, convert_to_tensor=True)
            cosine_scores = util.pytorch_cos_sim(query_embedding, result_embeddings)[0]

            # Combine scores with results and original indices
            scored_results = list(zip(cosine_scores.tolist(), original_indices, valid_results_with_text))
            
            # Sort primarily by score descending, secondarily by original index (stable sort for ties)
            scored_results.sort(key=lambda x: (x[0], -x[1]), reverse=True) # High score first, then low index (original order)

            # Return only the results, sorted
            reranked_list = [item[2] for item in scored_results]
            
            # Add back any results that had no text, placing them at the end
            results_without_text = [res for idx, res in enumerate(results) if idx not in original_indices]
            final_reranked_list = reranked_list + results_without_text

            if results and final_reranked_list and results[0] != final_reranked_list[0]:
                 print(f"  -> Semantic re-ranking changed top result.")
                 
            return final_reranked_list
            
        except Exception as e:
            print(f"  -> Error during semantic re-ranking: {e}. Returning original order.")
            return results

    # --- REVISED: run_web_search Method (Hybrid Approach) ---
    def run_web_search(self, analysis_result: dict, total_results_target: int = 10, max_pages_per_query: int = 3, results_per_page: int = 5) -> dict:
        """
        Performs web searches using a hybrid approach:
        1. Always queries SerpApi first for a small number of results.
        2. Supplements with NewsAPI (for news) or Tavily (for general/fallback) if needed.
        Handles pagination within the supplementary API call if necessary.

        Args:
            analysis_result: The dictionary returned by analyze_query.
            total_results_target: The desired total number of results across all sources.
            max_pages_per_query: Max number of pages to fetch *for the supplementary API*.
            results_per_page: How many results to request per *supplementary API* call page.

        Returns:
            A dictionary mapping each sub-query to its aggregated search results.
            The structure for each sub-query will be:
            {
                'api_used': list[str], # e.g., ['serpapi', 'newsapi']
                'results': list[dict], # List of combined & potentially re-ranked results
                'metadata': dict | None # Metadata primarily from the first *successful* API (SerpApi if it worked)
            }
        """
        search_results_collection = {}
        sub_queries = analysis_result.get("sub_queries", [])
        is_news_query = analysis_result.get("is_news_focused", False)
        INITIAL_SERPAPI_COUNT = 3 # Fetch this many from SerpApi initially

        if not sub_queries:
            print("No sub-queries found in analysis result to search for.")
            return search_results_collection

        print(f"\n--- Starting Hybrid Web Search Phase for {len(sub_queries)} sub-queries ---")
        print(f"Target total results: {total_results_target}, Initial SerpApi count: {INITIAL_SERPAPI_COUNT}")

        for query in sub_queries:
            final_results = []
            apis_successfully_used = [] # Track which APIs contributed
            primary_metadata = None # Store metadata from first success

            # --- 1. Initial SerpApi Call --- 
            serpapi_results_list = []
            if SERPAPI_API_KEY:
                print(f"Query: '{query[:60]}...' | Attempting initial SerpApi search (Count: {INITIAL_SERPAPI_COUNT})")
                try:
                    serpapi_response = perform_search(query, num_results=INITIAL_SERPAPI_COUNT, start_index=0)
                    if serpapi_response and serpapi_response.get('organic_results'):
                        serpapi_results_list = serpapi_response['organic_results']
                        final_results.extend(serpapi_results_list)
                        apis_successfully_used.append('serpapi')
                        primary_metadata = {k: v for k, v in serpapi_response.items() if k != 'organic_results'} # Store SerpApi metadata
                        print(f"  -> Found {len(serpapi_results_list)} initial results via SerpApi.")
                    elif serpapi_response:
                         print(f"  -> SerpApi call succeeded but returned no organic results.")
                         # Store metadata even if no results? Maybe only if error is not present.
                         if 'error' not in serpapi_response and primary_metadata is None:
                              primary_metadata = serpapi_response
                    else:
                         print(f"  -> Initial SerpApi call failed.")
                         # Keep primary_metadata as None
                except Exception as e:
                    print(f"  -> Error during initial SerpApi call: {e}")
            else:
                 print(f"  -> Skipping initial SerpApi call: Key not found.")

            # --- 2. Check if More Results Needed and Determine Supplementary API --- 
            supplementary_results_list = []
            if len(final_results) < total_results_target:
                remaining_target = total_results_target - len(final_results)
                print(f"  -> Need {remaining_target} more results. Determining supplementary API.")

                supplementary_api = None
                fallback_api = None # A final fallback if supplementary fails

                if is_news_query:
                    print("  -> News query detected. Prioritizing NewsAPI.")
                    if NEWS_API_KEY:
                         supplementary_api = 'newsapi'
                         fallback_api = 'tavily' if TAVILY_API_KEY else None
                    elif TAVILY_API_KEY: # News key missing, try Tavily directly
                         supplementary_api = 'tavily'
                         print("  -> NewsAPI key missing, using Tavily as supplementary.")
                    else:
                        print("  -> NewsAPI and Tavily keys missing. No supplementary search possible.")
                else: # General query
                    print("  -> General query detected. Prioritizing Tavily.")
                    if TAVILY_API_KEY:
                        supplementary_api = 'tavily'
                        fallback_api = 'newsapi' if NEWS_API_KEY else None
                    elif NEWS_API_KEY: # Tavily key missing, try NewsAPI directly
                        supplementary_api = 'newsapi'
                        print("  -> Tavily key missing, using NewsAPI as supplementary.")
                    else:
                        print("  -> Tavily and NewsAPI keys missing. No supplementary search possible.")

                # --- 3. Call Supplementary API (with Pagination/Fallback) ---
                current_api_to_try = supplementary_api
                attempt_fallback = True # Allow one fallback attempt if primary supplementary fails
                
                while current_api_to_try and len(supplementary_results_list) < remaining_target:
                    print(f"  -> Attempting supplementary search with {current_api_to_try} (Target: {remaining_target}).")
                    temp_results_this_api = []
                    api_metadata = None
                    
                    # Perform call(s) for the current supplementary API
                    try:
                        if current_api_to_try == 'newsapi':
                            # Simple pagination for NewsAPI within this block
                            current_page = 1
                            pages_fetched = 0
                            while len(temp_results_this_api) < remaining_target and pages_fetched < max_pages_per_query:
                                page_size = min(results_per_page, remaining_target - len(temp_results_this_api))
                                if page_size <= 0: break # Should not happen if loop condition is correct
                                news_response = get_news_articles(query, num_results=page_size, page=current_page)
                                pages_fetched += 1
                                if news_response and news_response.get('articles'):
                                    articles_found = news_response['articles']
                                    temp_results_this_api.extend(articles_found)
                                    if api_metadata is None: # Store metadata from first successful page
                                        api_metadata = {k: v for k, v in news_response.items() if k != 'articles'}
                                    current_page += 1
                                else:
                                    print(f"  -> No more results from NewsAPI on page {current_page} or call failed.")
                                    break # Stop NewsAPI pagination

                        elif current_api_to_try == 'tavily':
                            # Tavily doesn't paginate easily, make one call for remaining target
                            tavily_response = perform_tavily_search(query, max_results=remaining_target)
                            if tavily_response and tavily_response.get('results'):
                                temp_results_this_api = tavily_response['results']
                                api_metadata = {k: v for k, v in tavily_response.items() if k != 'results'} # Store Tavily metadata
                            else:
                                print(f"  -> Tavily call returned no results or failed.")

                    except Exception as e:
                        print(f"  -> Error during {current_api_to_try} call: {e}")
                        temp_results_this_api = [] # Ensure empty on error

                    # Process results from this API attempt
                    if temp_results_this_api:
                        supplementary_results_list.extend(temp_results_this_api)
                        apis_successfully_used.append(current_api_to_try)
                        if primary_metadata is None: # If SerpApi failed, use first successful supplementary metadata
                            primary_metadata = api_metadata
                        print(f"  -> Found {len(temp_results_this_api)} results via {current_api_to_try}. Total supplementary: {len(supplementary_results_list)}.")
                        # Stop trying APIs if we met the remaining target with this one
                        if len(supplementary_results_list) >= remaining_target:
                            break 
                    
                    # If this API failed/yielded nothing AND we can fallback
                    if not temp_results_this_api and attempt_fallback and fallback_api:
                        print(f"  -> {current_api_to_try} failed or insufficient. Trying fallback: {fallback_api}")
                        current_api_to_try = fallback_api
                        attempt_fallback = False # Only one fallback attempt
                    else:
                        # No more fallbacks or API succeeded (even if results < target)
                        break # Exit the supplementary API loop

            # --- 4. Combine and Re-rank Final Results ---
            final_results.extend(supplementary_results_list)
            final_results = final_results[:total_results_target]

            print(f"  -> Final combined results count before re-ranking: {len(final_results)}")
            
            # --- Re-ranking Step (Temporarily Disabled) ---
            if final_results and len(final_results) > 1: # Only attempt if multiple results
                # Original call (disabled):
                # final_results_ranked = self._rerank_search_results(final_results, analysis_result)
                
                # Bypass re-ranking for testing:
                final_results_ranked = final_results # Use un-ranked results
                print(f"  -> Skipping re-ranking (disabled for testing). Using {len(final_results_ranked)} results.")
            else: # No need to rerank if 0 or 1 results
                final_results_ranked = final_results
                print(f"  -> Skipping re-ranking (0 or 1 result). Using {len(final_results_ranked)} results.")

            # print(f"  -> Final combined results count after re-ranking: {len(final_results_ranked)}") # Keep commented out

            # --- 5. Store Results for Sub-query ---
            search_results_collection[query] = {
                'api_used': sorted(list(set(apis_successfully_used))), # Ensure consistent order 
                'results': final_results_ranked, # Use the potentially un-ranked results
                'metadata': primary_metadata
            }
            print(f"Finished processing query: '{query[:60]}...'. APIs Used: {apis_successfully_used}, Results: {len(final_results_ranked)}")

        print("\n--- Hybrid Web Search Phase Complete ---")
        print("Search Results Summary:")
        for q, data in search_results_collection.items():
             log_q = q if len(q) < 60 else q[:57] + "..."
             print(f"  - Query: '{log_q}', APIs: {data.get('api_used', [])}, Results Collected: {len(data.get('results', []))}")

        return search_results_collection

    # --- MODIFIED: Scrape/Crawl to handle new results structure ---
    def scrape_search_results(self, search_results_collection: dict, max_pages: int = 5, crawl_depth: int = 0, stay_on_domain: bool = True) -> dict:
        """
        Scrapes or crawls web pages, respecting robots.txt rules.
        Adapts to the aggregated results structure from run_web_search.

        Args:
            search_results_collection: Dict mapping sub-query to aggregated search results dict.
                                      (Structure: {'api_used': str, 'results': list[dict], 'metadata': dict|None})
            max_pages: Max number of unique pages to *scrape* or *crawl*.
            crawl_depth: How many link levels deep to crawl (0 = only initial URLs).
            stay_on_domain: If True, only crawls links within the same domain as the starting URL.

        Returns:
            A dictionary mapping visited URLs/keys to their extracted text content or direct answers/snippets.
        """
        scraped_content_collection = {}
        initial_urls = set()
        direct_content = {} # Store direct answers and snippets

        print(f"\n--- Aggregating URLs and Direct Content for Scraping ---")
        for sub_query, results_data in search_results_collection.items():
            log_q = sub_query if len(sub_query) < 60 else sub_query[:57] + "..."
            if not results_data:
                 print(f"  -> No results data found for query '{log_q}'. Skipping.")
                 continue

            # api_used is now a list, e.g., ['serpapi', 'newsapi']
            api_used_list = results_data.get('api_used', []) 
            results_list = results_data.get('results', [])
            metadata = results_data.get('metadata', {})

            # Handle direct answer from Tavily metadata (check if tavily was used)
            if 'tavily' in api_used_list and metadata and metadata.get('answer'):
                 answer_key = f'tavily_answer::{sub_query}'
                 direct_content[answer_key] = metadata['answer']
                 print(f"  -> Found direct Tavily answer for: {log_q}")

            # Extract URLs and snippets from the results list (handle mixed sources)
            urls_from_subquery = []
            snippets_added_count = 0
            if results_list:
                for result_item in results_list:
                    if not isinstance(result_item, dict):
                        continue # Skip non-dict items

                    url = None
                    snippet = None
                    title = result_item.get('title')

                    # --- UPDATED EXTRACTION LOGIC ---
                    # Try common keys regardless of specific API used
                    url = result_item.get('link') or result_item.get('url')
                    snippet = result_item.get('snippet') or result_item.get('content') or result_item.get('description')
                    # --- END UPDATED LOGIC ---

                    if url:
                        # Basic URL validation (optional)
                        parsed = urlparse(url)
                        if parsed.scheme in ['http', 'https']:
                            initial_urls.add(url)
                            urls_from_subquery.append(url)
                        else:
                            print(f"  -> Skipping invalid URL scheme: {url}")
                            url = None # Ensure invalid URL isn't used for snippet key

                        # Store snippet if available and url is valid
                        if snippet and url:
                            # Use a generic key, or try to guess source? Generic is safer.
                            snippet_key = f"snippet::{url}"
                            if snippet_key not in direct_content:
                                # Format snippet for clarity
                                direct_content[snippet_key] = f"Snippet: {str(snippet).strip()}"
                                snippets_added_count += 1

            print(f"  -> Processed {len(results_list)} results for '{log_q}'. APIs: {api_used_list}. Added {len(set(urls_from_subquery))} unique URLs. Added {snippets_added_count} snippets.")

        print(f"\n--- Starting Web Scraping/Crawling Phase ---")
        print(f"Initial unique URLs to scrape/crawl: {len(initial_urls)}")
        print(f"Items with direct content (Answers/Snippets): {len(direct_content)}")
        print(f"Max pages to visit (Scrape/Crawl): {max_pages}, Crawl depth: {crawl_depth}, Stay on domain: {stay_on_domain}")
        print(f"User-Agent for scraping: {USER_AGENT}")

        # --- Start Crawling/Scraping Process ---
        queue = deque([(url, 0) for url in initial_urls])
        visited = set(direct_content.keys())
        visited.update(initial_urls) 
        pages_attempted_scrape = 0
        crawl_results = {} 
        skipped_robots = 0

        print(f"Queue length for scraping: {len(queue)}")

        while queue and pages_attempted_scrape < max_pages:
            current_url, current_depth = queue.popleft()

            if current_url in crawl_results: # Already processed
                continue

            print(f"Processing URL (Depth {current_depth}): {current_url}")
            pages_attempted_scrape += 1

            # --- ROBOTS.TXT CHECK --- 
            robots_parser = self._get_robots_parser(current_url)
            if robots_parser and not robots_parser.is_allowed(USER_AGENT, current_url):
                 print(f"  -> Skipping disallowed by robots.txt: {current_url}")
                 skipped_robots += 1
                 crawl_results[current_url] = None # Mark as visited but not scraped
                 continue # Move to next URL in queue
            # --- END ROBOTS.TXT CHECK --- 
            
            # Scrape the current URL - EXPECTS STRUCTURED DICT OR NONE
            scraped_data = scrape_web_page(current_url, USER_AGENT)
            crawl_results[current_url] = scraped_data # Store dict or None

            # If crawling is enabled and content was scraped successfully
            # Check based on scraped_data being truthy (not None)
            if scraped_data and crawl_depth > 0 and current_depth < crawl_depth:
                print(f"  -> Finding links on {current_url} (Depth {current_depth})...")
                try:
                    # We need the HTML text to find links, get it from the dict
                    html_text = scraped_data.get('text', '')
                    if html_text:
                        soup = BeautifulSoup(html_text, 'lxml') # Assuming lxml is available
                        base_domain = urlparse(current_url).netloc
                        links_found_on_page = 0
                        for link in soup.find_all('a', href=True):
                            href = link['href']
                            next_url = urljoin(current_url, href)
                            parsed_next_url = urlparse(next_url)
                            # Normalize slightly for visited check
                            normalized_next_url = parsed_next_url._replace(fragment="").geturl()
                            next_domain = parsed_next_url.netloc

                            # Check if valid, not visited, and respects domain constraint
                            if normalized_next_url not in visited and \
                               parsed_next_url.scheme in ['http', 'https'] and \
                               (not stay_on_domain or next_domain == base_domain):
                                        # --- Check robots.txt for the potential next URL BEFORE adding ---
                                        next_robots_parser = self._get_robots_parser(normalized_next_url)
                                        if next_robots_parser and not next_robots_parser.is_allowed(USER_AGENT, normalized_next_url):
                                            # print(f"    -> Link disallowed by robots.txt, not adding to queue: {normalized_next_url}")
                                            continue # Skip adding this link
                                        # --- End Check ---

                                        queue.append((normalized_next_url, current_depth + 1))
                                        visited.add(normalized_next_url)
                                        links_found_on_page += 1
                        print(f"  -> Added {links_found_on_page} valid links to queue.")
                    else:
                        print(f"  -> No text content found in scraped data for link extraction.")
                except Exception as e:
                    print(f"  -> Error finding/processing links on {current_url}: {e}")

            # Politeness delay
            if queue and pages_attempted_scrape < max_pages:
                print(f"  -> Crawl delay ({CRAWL_DELAY}s)...")
                time.sleep(CRAWL_DELAY)


        print(f"--- Web Scraping/Crawling Phase Complete ---")
        print(f"Total unique pages visited (incl. direct content sources): {len(visited)}")
        print(f"Pages attempted scraping/crawling: {pages_attempted_scrape}")
        successful_scrapes = sum(1 for c in crawl_results.values() if c)
        print(f"Pages scraped successfully: {successful_scrapes}")

        # Combine direct content and crawled content
        final_scraped_content = direct_content.copy()
        # Add crawled results (structured dicts or None)
        final_scraped_content.update(crawl_results)

        return final_scraped_content # Returns dict mapping key -> (string | structured_dict | None)

    # --- Helper methods for formatting structured data (NEW) ---
    def _format_tables_for_prompt(self, tables: list) -> str:
        """Converts list of tables (list-of-lists) to Markdown format."""
        if not tables:
            return ""
        markdown = "\n\n--- PARSED TABLES ---\n"
        for i, table in enumerate(tables):
            markdown += f"\nTable {i+1}:\n"
            if not table: continue
            # Create header separator
            header = table[0]
            separator = ['---'] * len(header)
            # Format table using Markdown syntax
            markdown += f"| {' | '.join(map(str, header))} |\n"
            markdown += f"| {' | '.join(map(str, separator))} |\n"
            for row in table[1:]:
                markdown += f"| {' | '.join(map(str, row))} |\n"
            markdown += "\n"
        return markdown

    def _format_lists_for_prompt(self, lists: list) -> str:
        """Converts list of lists (list-of-strings) to Markdown format."""
        if not lists:
            return ""
        markdown = "\n\n--- PARSED LISTS ---\n"
        for i, lst in enumerate(lists):
            markdown += f"\nList {i+1}:\n"
            if not lst: continue
            for item in lst:
                markdown += f"- {str(item)}\n"
            markdown += "\n"
        return markdown

    # --- MODIFIED: analyze_content Method ---
    def analyze_content(self, scraped_data: dict | str | None, original_query: str, target_data_points: list[str] = None, is_news_focused: bool = False) -> dict:
        """
        Analyzes scraped data (text, tables, lists) for relevance and extracts info.

        Args:
            scraped_data: The structured data dict ('text', 'tables', 'lists') from 
                          scrape_web_page, or a simple string (for snippets/answers), or None.
            original_query: The initial query from the user.
            target_data_points: A list of specific data items to extract (optional).
            is_news_focused: Flag indicating if the original query was news-related.

        Returns:
            A dictionary containing the analysis:
            {
                "is_relevant": bool,
                "summary": str | None, # Summary if relevant, None otherwise
                "extracted_data": dict | None # Dictionary of {data_point: value} or None
            }
            Returns default values if LLM call fails.
        """
        analysis = {"is_relevant": False, "summary": None, "extracted_data": None} # Default result
        max_content_length = 15000 # Limit combined content length

        if not self.llm_client or not isinstance(self.llm_client, OpenAI):
            print("Cannot analyze content: LLM client not available or not OpenAI.")
            return analysis

        # --- Extract and Format Content --- 
        text_content = ""
        formatted_tables = ""
        formatted_lists = ""
        
        if isinstance(scraped_data, dict):
            text_content = scraped_data.get('text', '') or "" # Ensure string
            tables = scraped_data.get('tables', [])
            lists = scraped_data.get('lists', [])
            formatted_tables = self._format_tables_for_prompt(tables)
            formatted_lists = self._format_lists_for_prompt(lists)
        elif isinstance(scraped_data, str): # Handle direct answers/snippets
            text_content = scraped_data
        else: # Handle None or other unexpected types
             print("Cannot analyze content: Input data is empty or invalid type.")
             analysis['error'] = "Invalid or empty content provided."
             return analysis

        # Combine content for the prompt
        combined_content = text_content + formatted_tables + formatted_lists
        if not combined_content.strip():
             print("Cannot analyze content: Combined text, tables, and lists are empty.")
             analysis['error'] = "No analyzable content found."
             return analysis

        # Truncate combined content if too long
        truncated_content = combined_content[:max_content_length]
        if len(combined_content) > max_content_length:
             print(f"Warning: Combined content truncated to {max_content_length} characters for analysis.")

        if target_data_points is None:
            target_data_points = []

        # --- Modify System Prompts to mention structured data ---
        base_system_prompt_template = """You are an expert content analyzer for a web research agent.
        Your task is to determine if the provided text content is relevant to the user's original query,
        provide a concise summary focusing *only* on the aspects relevant to the query, and extract specific data points if requested.

        Evaluate the following CONTENT (which may include main text, parsed tables, and parsed lists) based on its relevance to the ORIGINAL QUERY.
        {data_extraction_instruction}
        Return your analysis STRICTLY as a JSON object with the following keys:
        - "is_relevant": boolean (true if the text directly addresses or provides significant information about the original query, false otherwise).
        - "summary": string (If relevant, provide a concise summary (2-4 sentences) of the information in the text that directly relates to the original query. If not relevant, this should be null or an empty string).
        - "extracted_data": object (If specific data points were requested, return a JSON object mapping each requested data point to its found value in the text (as string). If a value is not found, use null. If no data points were requested, return an empty object {{}}).

        Focus only on the provided content. Use the parsed tables and lists if they help answer the query or extract data.
        """

        news_system_prompt_template = """You are an expert content analyzer specializing in news articles for a web research agent.
        Your task is to determine if the provided text content, likely from a news source, is relevant to the user's original query asking about recent events or news.
        If relevant, provide a concise news-style summary and extract specific data points if requested.

        Evaluate the following CONTENT (which may include main text, parsed tables, and parsed lists) based on its relevance to the ORIGINAL QUERY, assuming it's potentially a news report.
        {data_extraction_instruction}
        Return your analysis STRICTLY as a JSON object with the following keys:
        - "is_relevant": boolean (true if the text directly addresses or provides significant information about the original query, false otherwise).
        - "summary": string (If relevant, provide a concise summary (2-4 sentences) focusing on key news elements like *what happened, who was involved, key outcomes, date/timeframe if mentioned*. If not relevant, this should be null or an empty string).
        - "extracted_data": object (If specific data points were requested, return a JSON object mapping each requested data point to its found value in the text (as string). If a value is not found, use null. If no data points were requested, return an empty object {{}}).

        Focus only on the provided content. Prioritize information directly related to the news aspect of the query. If the text is relevant but not a news report, summarize its relevant points normally.
        """

        # Prepare data extraction instruction part of the prompt
        data_extraction_instruction = ""
        if target_data_points:
            data_points_str = ", ".join([f'"{dp}"' for dp in target_data_points])
            data_extraction_instruction = f"Additionally, extract the specific values for the following data points if present in the text, tables, or lists: [{data_points_str}]."
        else:
            data_extraction_instruction = "No specific data points requested for extraction."

        prompt_template = news_system_prompt_template if is_news_focused else base_system_prompt_template
        system_prompt = prompt_template.format(data_extraction_instruction=data_extraction_instruction)
        prompt_type = "news-focused" if is_news_focused else "standard"

        # Use the truncated combined content in the user prompt
        user_prompt = f"ORIGINAL QUERY: \"{original_query}\"\n\nTARGET DATA POINTS: {target_data_points}\n\nCONTENT:\n{truncated_content}"

        try:
            print(f"Analyzing content relevance ({prompt_type} prompt) for query: '{original_query[:50]}...'")
            response = self.llm_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1, 
                response_format={"type": "json_object"}
            )
            response_content = response.choices[0].message.content
            if response_content:
                parsed_analysis = json.loads(response_content)
                required_keys = {"is_relevant", "summary", "extracted_data"}
                if required_keys.issubset(parsed_analysis.keys()):
                    analysis["is_relevant"] = bool(parsed_analysis["is_relevant"])
                    analysis["summary"] = parsed_analysis["summary"] if analysis["is_relevant"] and parsed_analysis["summary"] else None
                    extracted_data = parsed_analysis.get("extracted_data")
                    # Ensure extracted data is stored as dict if relevant, otherwise None
                    if analysis["is_relevant"] and isinstance(extracted_data, dict):
                        analysis["extracted_data"] = extracted_data
                    else:
                        analysis["extracted_data"] = None
                    print(f"Content analysis successful. Relevant: {analysis['is_relevant']}")
                else:
                     print("Error: Content analysis LLM response missing required keys.")
                     analysis['error'] = "LLM response missing keys"
            else:
                print("Error: Content analysis LLM returned empty content.")
                analysis['error'] = "LLM returned empty content"

        except APIError as e:
            print(f"OpenAI API Error during content analysis: {e}")
            analysis['error'] = f"API Error: {getattr(e.response, 'status_code', 'N/A')}"
        except json.JSONDecodeError as e:
             print(f"Error parsing content analysis LLM JSON response: {e}")
             analysis['error'] = "LLM response JSON parsing error"
        except Exception as e:
            print(f"Unexpected error during LLM call for content analysis: {e}")
            analysis['error'] = f"Unexpected analysis error: {type(e).__name__}"

        return analysis

    def synthesize_results(self, original_query: str, content_analysis: dict) -> str:
        """
        Synthesizes the analyzed content summaries into a final report.

        Args:
            original_query: The initial query from the user.
            content_analysis: Dictionary mapping URLs to analysis results
                              ({'is_relevant': bool, 'summary': str|None}).

        Returns:
            A string containing the synthesized report, or an error/empty message.
        """
        report = "Synthesis failed: No relevant content found or LLM error."
        max_summary_length = 10000 # Limit total summary length sent to LLM

        if not self.llm_client or not isinstance(self.llm_client, OpenAI):
            print("Cannot synthesize results: LLM client not available or not OpenAI.")
            return report

        # Filter relevant summaries and extracted data
        relevant_content = []
        relevant_sources = []
        for url, analysis_data in content_analysis.items():
            if analysis_data and analysis_data.get('is_relevant'):
                source_info = {
                    "url": url,
                    "summary": analysis_data.get('summary'),
                    "extracted_data": analysis_data.get('extracted_data')
                }
                # Only include if there's a summary OR extracted data
                if source_info["summary"] or (isinstance(source_info["extracted_data"], dict) and source_info["extracted_data"]):
                    relevant_content.append(source_info)
                relevant_sources.append(url)

        if not relevant_content:
            print("No relevant summaries or extracted data found to synthesize.")
            return "Synthesis failed: No relevant content or data points were generated from the sources."

        # Combine content for the prompt, respecting length limit
        content_input_str = ""
        for item in relevant_content:
            content_input_str += f"Source URL: {item['url']}\n"
            if item['summary']:
                content_input_str += f"Summary: {item['summary']}\n"
            if item['extracted_data']:
                # Pretty print dict for clarity in prompt
                data_str = json.dumps(item['extracted_data'], indent=2)
                content_input_str += f"Extracted Data: {data_str}\n"
            content_input_str += "---\n"

        truncated_content_input = content_input_str[:max_summary_length] # Use the same limit
        if len(content_input_str) > max_summary_length:
            print(f"Warning: Combined content input truncated to {max_summary_length} characters for synthesis.")

        system_prompt = """You are an expert research report synthesizer.
        Your task is to compile the provided content summaries AND extracted data points into a single, coherent, and well-structured report that directly answers the user's original query.

        - Start with a brief introductory sentence setting the context based on the original query.
        - Use the information presented in the CONTENT FROM SOURCES section to build the body of the report.
        - Synthesize findings, identify key points, and structure the information logically.
        - Integrate both the summaries and the specific extracted data points naturally into your response. If extracted data points are available, prioritize using them to answer relevant parts of the query.
        - If different sources present conflicting information (either in summaries or extracted data), acknowledge the discrepancy if significant.
        - Conclude with a brief concluding sentence.
        - Write in a clear, objective, and informative tone.
        - Do NOT include information not present in the provided content.
        - Do NOT invent new information or draw unsupported conclusions.
        - Aim for a concise yet comprehensive report.
        - Do not explicitly mention 'the summaries provided' or 'extracted data'. Integrate the information naturally.
        """

        # Use a single multi-line f-string for clarity
        user_prompt = f"""ORIGINAL QUERY: "{original_query}"

CONTENT FROM RELEVANT SOURCES:
{truncated_content_input}
Based ONLY on the content above, generate a synthesized report answering the original query."""

        try:
            print(f"Synthesizing report for query: '{original_query[:50]}...'")
            response = self.llm_client.chat.completions.create(
                # Consider using a more capable model like gpt-4 if needed for complex synthesis
                model=self.model_name, 
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5, # Allow for some creativity in synthesis structure
                # No specific response format needed, just text
            )
            response_content = response.choices[0].message.content
            if response_content:
                report = response_content.strip()
                print(f"Report synthesis successful.")
            else:
                print("Error: Synthesis LLM returned empty content.")
                report = "Synthesis failed: LLM returned empty content."

        except APIError as e:
            print(f"OpenAI API Error during synthesis: {e}")
            report = f"Synthesis failed: OpenAI API Error {e.status_code}"
        except Exception as e:
            print(f"Unexpected error during LLM call for synthesis: {e}")
            report = f"Synthesis failed: Unexpected error ({type(e).__name__})"

        return report

    # --- MODIFIED: Orchestrating Method ---
    def run_research_pipeline(self, user_query: str, total_results_target: int = 10, max_pages_to_crawl: int = 5, crawl_depth: int = 0) -> dict:
        """Runs the full research pipeline: query analysis, web search (multi-tool w/ pagination), scraping/crawling, content analysis, synthesis."""
        final_results = {
            "query_analysis": None,
            "search_results": None,
            "scraped_content": None,
            "content_analysis": {},
            "synthesized_report": "Pipeline did not complete." # Default message
        }
        print(f"Starting research pipeline for query: {user_query}")

        # 1. Analyze Query
        query_analysis = self.analyze_query(user_query)
        final_results["query_analysis"] = query_analysis
        if not query_analysis or query_analysis.get("intent") == "Analysis failed" or not query_analysis.get("sub_queries"):
            print("Pipeline stopped: Query analysis failed or produced no sub-queries.")
            final_results["synthesized_report"] = "Pipeline stopped before synthesis due to query analysis failure."
            return final_results

        # 2. Run Web Search (Now uses pagination)
        # Define results_per_page for the API calls within run_web_search
        RESULTS_PER_SEARCH_PAGE = 5 # How many items to fetch per API call page
        MAX_SEARCH_PAGES = 3 # Max API pages to fetch per sub-query

        search_results = self.run_web_search(
            query_analysis,
            total_results_target=total_results_target,
            max_pages_per_query=MAX_SEARCH_PAGES,
            results_per_page=RESULTS_PER_SEARCH_PAGE
        )
        final_results["search_results"] = search_results
        # Check if ANY search across ANY sub-query returned results
        if not search_results or all(not data.get('results') for data in search_results.values()):
             print("Pipeline stopped: Web search failed or produced no results from any API for any sub-query.")
             final_results["synthesized_report"] = "Pipeline stopped before synthesis due to web search failure (no results found)."
             return final_results

        # 3. Determine Crawl Strategy Based on Analysis
        DEFAULT_MAX_PAGES_NO_CRAWL = 5 # Max pages to *scrape* if no crawl
        DEFAULT_MAX_PAGES_WITH_CRAWL = 10 # Max pages to *scrape/crawl* if crawl enabled
        CRAWL_DEPTH_IF_ENABLED = 1

        if query_analysis.get("target_data_points"):
             print("Query requires specific data points. Enabling shallow crawl (Depth 1).")
             crawl_depth_to_use = CRAWL_DEPTH_IF_ENABLED
             max_scrape_pages_to_use = DEFAULT_MAX_PAGES_WITH_CRAWL
        else:
             print("Query is general. Scraping initial results only (Depth 0).")
             crawl_depth_to_use = 0
             max_scrape_pages_to_use = DEFAULT_MAX_PAGES_NO_CRAWL

        # Override max_pages_to_crawl if provided directly
        if max_pages_to_crawl is not None:
             max_scrape_pages_to_use = max_pages_to_crawl
             print(f"Using provided max_pages_to_crawl: {max_scrape_pages_to_use}")


        # 4. Scrape/Crawl Results (Using updated scrape_search_results)
        scraped_content = self.scrape_search_results(
            search_results,
            max_pages=max_scrape_pages_to_use, # Use the determined limit for scraping/crawling
            crawl_depth=crawl_depth_to_use,
            stay_on_domain=True
        )
        final_results["scraped_content"] = scraped_content
        if not scraped_content:
             # Don't stop here, maybe direct answers/snippets were enough
             print("Pipeline continued: Web scraping/crawling produced no new content beyond initial snippets/answers.")
             # Allow synthesis to proceed with only direct_content if available

        # 5. Analyze Content for Relevance and Summarize
        content_analysis_results = {}
        content_to_analyze = final_results.get("scraped_content", {})
        target_data = query_analysis.get("target_data_points", [])
        is_news = query_analysis.get("is_news_focused", False)
        items_analyzed_count = 0

        print(f"\n--- Starting Content Analysis Phase ({len(content_to_analyze)} items) ---")
        for item_key, text in content_to_analyze.items():
            source_type = "unknown" # Default
            display_key = item_key # Default
            analysis_result = None # Store result here
            error_message = None # Store potential error message

            # 1. Identify the type of item
            if item_key.startswith('tavily_answer::'):
                source_type = "tavily_answer"
                display_key = f"Tavily Answer ({item_key.split('::')[1][:30]}...)"
                # Directly use the answer text as summary, mark as relevant
                if text:
                    # Assign analysis result directly for Tavily answers
                    analysis_result = {"is_relevant": True, "summary": text, "extracted_data": None, "source_type": source_type}
                    print(f"Using direct {display_key}")
                    # We count this as 'analyzed' as it contributes to the report
                    items_analyzed_count += 1
                else:
                    error_message = "Tavily answer content missing."
                    print(f"Warning: Missing content for {display_key}")

            elif item_key.startswith(('newsapi_snippet::', 'tavily_snippet::', 'serpapi_snippet::')):
                source_type = item_key.split('_snippet::')[0] + "_snippet"
                original_url = item_key.split('::')[1]
                display_key = f"Snippet from {source_type} for {original_url}"
                # Snippets will be analyzed if text exists; set error if not
                if not text:
                    error_message = "Snippet content missing."
                    print(f"Skipping analysis for {display_key} (no snippet content)")

            else: # Assumed to be a scraped URL (or failed scrape)
                source_type = "scraped_page"
                display_key = f"Scraped Page: {item_key}"
                # Set error if no text (scrape failed)
                if not text:
                    error_message = "Scraping failed or returned empty content."
                    display_key = f"Failed Scrape: {item_key}" # Update display key for error case
                print(f"Skipping analysis for {display_key} (no content)")

            # 2. Analyze content if it exists and hasn't been handled directly (like Tavily answer) or marked as error
            if text and not analysis_result and not error_message:
                print(f"Analyzing {display_key} ({source_type})")
                analysis = self.analyze_content(text, user_query, target_data, is_news)
                analysis["source_type"] = source_type # Add source type to analysis result
                analysis_result = analysis
                items_analyzed_count += 1 # Increment only when LLM analysis is performed

            # 3. Store the final result or error for this item
            if analysis_result:
                 # Store the successfully generated analysis (either direct or from LLM)
                 content_analysis_results[item_key] = analysis_result
            elif error_message:
                 # Store an error entry if content was missing or scrape failed
                 content_analysis_results[item_key] = {"is_relevant": False, "summary": None, "extracted_data": None, "error": error_message, "source_type": source_type}
            # else: If text existed but analyze_content failed internally, it returns a default dict which gets stored via analysis_result


        # This should align with the `for` loop
        final_results["content_analysis"] = content_analysis_results
        print(f"--- Content Analysis Phase Complete (Analyzed {items_analyzed_count} items) ---")
        
        # 6. Synthesize Final Report
        synthesized_report = self.synthesize_results(user_query, content_analysis_results)
        final_results["synthesized_report"] = synthesized_report

        print("\n--- Research Pipeline Complete ---")
        return final_results

# --- Updated Example Usage (Fixing try/except) --- 
if __name__ == '__main__':
    agent = WebResearchAgent()
    if not agent.llm_client:
        print("\nLLM Client failed to initialize. Exiting.")
    else:
        # --- Test Case 1 (General Query) ---
        print("\n" + "="*40)
        print("--- Running General Query Test (Pagination) --- ")
        print("="*40)
        query1 = "Benefits of using FastAPI vs Flask"
        print(f"Running Query 1: {query1}")
        try:
            results1 = agent.run_research_pipeline(query1, total_results_target=8)
            print("\nSynthesized Report (General Query):")
            print(results1.get("synthesized_report", "No report generated."))
        except Exception as e_main: # Ensure except clause exists and is indented
            print(f"\n*** Error running pipeline for query 1: {e_main} ***")
            import traceback
            traceback.print_exc()

        # --- Test Case 2 (News Query) ---
        print("\n" + "="*40)
        print("--- Running News Query Test (Pagination) --- ")
        print("="*40)
        query2 = "Latest developments in renewable energy technology"
        print(f"Running Query 2: {query2}")
        try:
            results2 = agent.run_research_pipeline(query2, total_results_target=7)
            print("\nSynthesized Report (News Query):")
            print(results2.get("synthesized_report", "No report generated."))
        except Exception as e_main: # Ensure except clause exists and is indented
            print(f"\n*** Error running pipeline for query 2: {e_main} ***")
            import traceback
            traceback.print_exc()

        # --- Test Case 3 (Specific Data Query) ---
        print("\n" + "="*40)
        print("--- Running Specific Data + Crawl Test --- ")
        print("="*40)
        query3 = "What is the battery capacity and charging speed of the Ford Mustang Mach-E GT?"
        print(f"Running Query 3: {query3}")
        try:
            results3 = agent.run_research_pipeline(query3, total_results_target=5)
            print("\nSynthesized Report (Specific Data Query):")
            print(results3.get("synthesized_report", "No report generated."))
        except Exception as e_main: # Ensure except clause exists and is indented
            print(f"\n*** Error running pipeline for query 3: {e_main} ***")
            import traceback
            traceback.print_exc() 