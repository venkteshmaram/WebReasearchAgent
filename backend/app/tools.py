import os
import time
import requests
from bs4 import BeautifulSoup
from serpapi import GoogleSearch
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from collections import deque
from urllib.parse import urljoin, urlparse
from newsapi import NewsApiClient
from tavily import TavilyClient
from robotexclusionrulesparser import RobotExclusionRulesParser
import pandas as pd
from io import StringIO

# Load environment variables (like API keys)
load_dotenv()

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# --- Mock Browser Headers ---
HEADERS = {
    'User-Agent': 'WebResearchAgent/1.0',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'none',
    'Sec-Fetch-User': '?1',
    'Cache-Control': 'max-age=0',
}

REQUEST_TIMEOUT = 30 # seconds (Increased from 15)
MIN_CONTENT_LENGTH = 100 # Minimum characters to consider BS4 successful
SELENIUM_WAIT_TIME = 10 # seconds to wait for JS loading in Selenium (Increased from 5)
CRAWL_DELAY = 1 # seconds between requests during crawl

if not SERPAPI_API_KEY:
    print("Warning: SERPAPI_API_KEY not found. Real web search will fail.")
if not NEWS_API_KEY:
    print("Warning: NEWS_API_KEY not found. News searches will fail.")
if not TAVILY_API_KEY:
    print("Warning: TAVILY_API_KEY not found. Tavily search will fail.")

# --- Robots.txt Checker --- 
ROBOTS_CACHE = {} # Simple in-memory cache for robots.txt parsers
ROBOTS_TIMEOUT = 5 # Shorter timeout for fetching robots.txt

def is_scraping_allowed(target_url: str, user_agent: str) -> bool:
    """Checks if scraping the target_url is allowed based on robots.txt.

    Args:
        target_url: The URL to check.
        user_agent: The User-Agent string of the scraper.

    Returns:
        True if allowed (or robots.txt not found/parse failed), False if disallowed.
    """
    try:
        parsed_url = urlparse(target_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = f"{base_url}/robots.txt"

        # Check cache first
        if base_url in ROBOTS_CACHE:
            parser = ROBOTS_CACHE[base_url]
            if parser is None: # Indicates previous fetch failure
                # print(f"Robots.txt check skipped for {target_url} (previous fetch failed).")
                return True 
            # print(f"Using cached robots.txt parser for {base_url}")
            is_allowed = parser.is_allowed(user_agent, target_url)
            # print(f"Robots.txt check for {target_url}: {'Allowed' if is_allowed else 'Disallowed'}")
            return is_allowed

        # Fetch robots.txt if not cached
        parser = None # Default to None (allow if fetch fails)
        try:
            print(f"Fetching robots.txt from: {robots_url}")
            response = requests.get(robots_url, headers={'User-Agent': user_agent}, timeout=ROBOTS_TIMEOUT, allow_redirects=True)
            
            if response.status_code == 200:
                parser = RobotExclusionRulesParser()
                parser.parse(response.text)
                print(f"Successfully fetched and parsed robots.txt for {base_url}")
            elif response.status_code == 404:
                print(f"No robots.txt found for {base_url} (404). Assuming allowed.")
                # No rules means allowed, store None to indicate no rules found (or explicit allow)
                parser = None # Or perhaps a parser that always allows?
            else:
                print(f"Failed to fetch robots.txt for {base_url} (Status: {response.status_code}). Assuming allowed.")
                parser = None # Allow on server errors

        except requests.exceptions.Timeout:
            print(f"Timeout fetching robots.txt for {base_url}. Assuming allowed.")
            parser = None
        except requests.exceptions.RequestException as e:
            print(f"Error fetching robots.txt for {base_url}: {e}. Assuming allowed.")
            parser = None
        except Exception as e:
             print(f"Error parsing robots.txt for {base_url}: {e}. Assuming allowed.")
             parser = None # Assume allowed if parsing fails
             
        # Store the result (parser or None) in cache
        ROBOTS_CACHE[base_url] = parser
        
        if parser is None:
             return True # Assume allowed if fetch/parse failed or 404

        is_allowed = parser.is_allowed(user_agent, target_url)
        # print(f"Robots.txt check for {target_url}: {'Allowed' if is_allowed else 'Disallowed'}")
        return is_allowed

    except Exception as e:
        print(f"Unexpected error during robots.txt check for {target_url}: {e}. Assuming allowed.")
        return True # Fail open

def perform_search(query: str, num_results: int = 10, start_index: int = 0) -> dict | None:
    """
    Performs a web search using SerpApi.

    Args:
        query: The search query string.
        num_results: The desired number of results per page (default is 10).
        start_index: The starting index for results (for pagination, default is 0).

    Returns:
        A dictionary containing the search results or None if failed.
    """
    if not SERPAPI_API_KEY:
        print("Error: Cannot perform search, SERPAPI_API_KEY is missing.")
        return None

    params = {
        "q": query,
        "num": num_results,
        "start": start_index, # Added for pagination
        "api_key": SERPAPI_API_KEY,
        "engine": "google",  # Specify the search engine
        # Add other parameters as needed, e.g.:
        # "location": "Austin, Texas",
        # "hl": "en",
        # "gl": "us",
    }

    try:
        print(f"Performing SerpApi search for: '{query}' (Start Index: {start_index})") 
        search = GoogleSearch(params)
        results = search.get_dict()

        if "error" in results:
            print(f"SerpApi Error (explicit error key): {results['error']}")
            return None

        # Basic check if organic results exist
        if not results.get("organic_results"):
             print(f"No organic results found for query: '{query}' on this page (Start Index: {start_index}).")
             # --- ADDED DEBUG PRINT --- 
             print(f"Full SerpApi Response when no organic results:\n{results}")
             # --- END DEBUG PRINT --- 
             # Still return results dict, might contain pagination info or other data

        print(f"SerpApi Search successful for: '{query}' (Start Index: {start_index})")
        return results

    except Exception as e:
        print(f"An error occurred during SerpApi search: {e}")
        return None

# --- NEW: News API Function ---
def get_news_articles(query: str, num_results: int = 10, page: int = 1) -> dict | None:
    """
    Fetches news articles using the News API.

    Args:
        query: The search query/keywords for news articles.
        num_results: The desired number of articles per page (page size). Max 100.
        page: The page number to fetch (for pagination).

    Returns:
        A dictionary containing the API response (articles, status, etc.)
        or None if the search failed or the API key is missing.
    """
    if not NEWS_API_KEY:
        print("Error: Cannot perform news search, NEWS_API_KEY is missing.")
        return None

    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        # Use page_size for num_results per page, max 100
        page_size = min(num_results, 100)
        print(f"Performing NewsAPI search for: '{query}' (Page: {page}, PageSize: {page_size})") # Added page info

        # Option 1: Top headlines (more restrictive on free plan)
        # top_headlines = newsapi.get_top_headlines(q=query, language='en', page_size=page_size, page=page) # Added page
        # if top_headlines.get('status') == 'ok':
        #     print(f"NewsAPI Search successful (Top Headlines) for: '{query}' (Page {page})")
        #     return top_headlines
        # print(f"NewsAPI Top Headlines failed, trying everything search...")

        # Option 2: Everything search (more flexible on free plan, but might be older)
        all_articles = newsapi.get_everything(q=query,
                                              language='en',
                                              sort_by='relevancy', # or 'publishedAt'
                                              page_size=page_size,
                                              page=page) # Added page parameter

        if all_articles.get('status') == 'ok':
            print(f"NewsAPI Search successful ('Everything') for: '{query}' (Page {page})")
            # Check if articles list is empty for this page
            if not all_articles.get("articles"):
                print(f"No articles found for query: '{query}' on Page {page}.")
            # No need to slice here, the API call fetches the specific page
            return all_articles
        else:
            error_message = all_articles.get('message', 'Unknown NewsAPI error')
            print(f"NewsAPI Error: {error_message}")
            # Check for specific rate limit or other errors if needed
            return None

    except Exception as e:
        print(f"An error occurred during NewsAPI search: {e}")
        return None

# --- NEW: Tavily Search Function ---
def perform_tavily_search(query: str, search_depth: str = "basic", max_results: int = 5) -> dict | None:
    """
    Performs a web search optimized for LLMs using Tavily.

    Args:
        query: The search query string.
        search_depth: Level of search depth ("basic" or "advanced").
        max_results: The desired number of results.

    Returns:
        A dictionary containing the search results from Tavily
        or None if the search failed or the API key is missing.
    """
    if not TAVILY_API_KEY:
        print("Error: Cannot perform Tavily search, TAVILY_API_KEY is missing.")
        return None

    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        print(f"Performing Tavily search (Depth: {search_depth}) for: '{query}'")
        response = client.search(query=query, search_depth=search_depth, max_results=max_results)
        # Response structure includes: query, response_time, answer, results (list of dicts)
        print(f"Tavily Search successful for: '{query}'")
        return response # Return the full Tavily response dictionary

    except Exception as e:
        print(f"An error occurred during Tavily search: {e}")
        return None

# --- NEW: Web Scraper Function --- 

def _table_to_markdown(table_soup: BeautifulSoup) -> str:
    """Converts a BeautifulSoup table object to a Markdown string.
    Uses pandas for robust parsing.
    """
    try:
        # Use pandas read_html which is good at handling complex tables
        # StringIO is used to treat the table string as a file
        dfs = pd.read_html(StringIO(str(table_soup)), header=0) # Assume first row is header
        if not dfs:
            return "" # No table found by pandas
        # We usually want the first table found within the soup element
        df = dfs[0]
        # Convert DataFrame to Markdown format, index=False to avoid row numbers
        markdown_table = df.to_markdown(index=False)
        return f"\n\n{markdown_table}\n\n" # Add padding for clarity
    except Exception as e:
        # Fallback or error logging if pandas fails
        print(f"Note: Could not parse table with pandas: {e}. Skipping table.")
        # Simple fallback (less robust):
        # rows = []
        # for tr in table_soup.find_all('tr'):
        #     cells = [cell.get_text(strip=True) for cell in tr.find_all(['td', 'th'])]
        #     rows.append(f"| {' | '.join(cells)} |")
        # if not rows: return ""
        # # Basic Markdown structure
        # header_sep = f"| {' | '.join(['---' for _ in rows[0].split('|')[1:-1]])} |"
        # return f"\n{rows[0]}\n{header_sep}\n" + '\n'.join(rows[1:]) + "\n"
        return "" # Return empty string on error

def _extract_text_from_soup(soup: BeautifulSoup) -> str:
    """Extracts meaningful text from a BeautifulSoup object, removing script/style."""
    # Remove script and style elements
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()

    # Get text, strip whitespace, and join lines
    lines = (line.strip() for line in soup.get_text().splitlines())
    # Remove empty lines and join
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = '\n'.join(chunk for chunk in chunks if chunk)
    return text

def scrape_web_page(url: str, user_agent: str) -> dict | None:
    """Scrapes text, tables, and lists from a URL using Requests and BeautifulSoup."""
    print(f"  -> Scraping URL: {url}")
    headers = {
        'User-Agent': user_agent
    }
    try:
        # Use requests library
        response = requests.get(url, headers=headers, timeout=10) 
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        # Check content type - only parse HTML
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            print(f"  -> Skipping non-HTML content type: {content_type}")
            return None # Indicate non-HTML content

        soup = BeautifulSoup(response.text, 'lxml')

        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()

        # Extract main text content, trying common tags
        text_content = ''
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        if main_content:
            # Get text chunks and join, separating paragraphs
            text_chunks = main_content.find_all(string=True, recursive=True)
            text_content = '\n\n'.join(chunk.strip() for chunk in text_chunks if chunk.strip())
        
        # Extract tables (as list of lists)
        tables_data = []
        for table in soup.find_all('table'):
            table_rows = []
            for row in table.find_all('tr'):
                # Get headers and data cells
                cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                if cells:
                    table_rows.append(cells)
            if table_rows:
                tables_data.append(table_rows)

        # Extract lists (as list of lists)
        lists_data = []
        for lst in soup.find_all(['ul', 'ol']):
            list_items = [item.get_text(strip=True) for item in lst.find_all('li', recursive=False) if item.get_text(strip=True)]
            if list_items:
                lists_data.append(list_items)
        
        if not text_content and not tables_data and not lists_data:
            print(f"  -> No relevant content (text, tables, lists) found on {url}")
            return None # Indicate no useful content extracted

        print(f"  -> Successfully scraped {url}")
        return {
            'url': url,
            'text': text_content,
            'tables': tables_data,
            'lists': lists_data
        }

    except requests.exceptions.RequestException as e:
        print(f"  -> Error scraping {url}: {e}")
        return None # Indicate scraping error
    except Exception as e:
        print(f"  -> Unexpected error processing {url}: {e}")
        return None

# --- NEW: Scrape/Crawl Search Results Method ---
def scrape_search_results(
    search_results_collection: dict,
    max_pages: int = 5,
    crawl_depth: int = 0, # 0 means only scrape initial URLs, 1 means scrape initial + links found on them, etc.
    stay_on_domain: bool = True
) -> dict:
    """
    Scrapes or crawls web pages found in the search results.

    Args:
        search_results_collection: Dict mapping sub-query to SerpApi results dict.
        max_pages: The maximum total number of unique pages to scrape/crawl.
        crawl_depth: How many link levels deep to crawl (0 = only initial URLs).
        stay_on_domain: If True, only crawls links within the same domain as the starting URL.

    Returns:
        A dictionary mapping visited URLs to their extracted text content (or None if failed).
    """
    scraped_content_collection = {}
    initial_urls = set() # Use a set to get unique initial URLs

    # Gather unique initial URLs from organic results across all sub-queries
    for sub_query, results_data in search_results_collection.items():
        if results_data and "organic_results" in results_data:
            for result in results_data["organic_results"]:
                url = result.get('link')
                if url:
                    initial_urls.add(url)

    print(f"\n--- Starting Web Scraping/Crawling Phase ---")
    print(f"Initial unique URLs from search: {len(initial_urls)}")
    print(f"Max pages to visit: {max_pages}, Crawl depth: {crawl_depth}, Stay on domain: {stay_on_domain}")

    queue = deque([(url, 0) for url in initial_urls]) # Queue of (url, depth)
    visited = set() # Keep track of visited URLs to avoid loops and re-scraping
    pages_scraped_count = 0

    while queue and pages_scraped_count < max_pages:
        current_url, current_depth = queue.popleft()

        # Normalize URL slightly (optional, but can help with duplicates)
        parsed_url = urlparse(current_url)
        normalized_url = parsed_url._replace(fragment="").geturl() # Remove fragments

        if normalized_url in visited:
            continue

        print(f"Processing URL (Depth {current_depth}): {normalized_url}")
        visited.add(normalized_url)

        # Scrape the current URL
        content = scrape_web_page(normalized_url, HEADERS['User-Agent']) # Use the existing robust scraper
        scraped_content_collection[normalized_url] = content
        pages_scraped_count += 1

        # If crawling is enabled (depth > 0) and content was scraped successfully
        if content and crawl_depth > 0 and current_depth < crawl_depth:
            print(f"  -> Finding links on {normalized_url} (Depth {current_depth})...")
            try:
                soup = BeautifulSoup(content['text'], 'lxml') # Re-parse text to find links
                base_domain = urlparse(normalized_url).netloc

                links_found_on_page = 0
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    # Construct absolute URL
                    next_url = urljoin(normalized_url, href)
                    # Normalize again
                    parsed_next_url = urlparse(next_url)
                    normalized_next_url = parsed_next_url._replace(fragment="").geturl()
                    next_domain = parsed_next_url.netloc

                    # Basic validation and filtering
                    if normalized_next_url not in visited and \
                       parsed_next_url.scheme in ['http', 'https'] and \
                       (not stay_on_domain or next_domain == base_domain):
                           # Check if adding this exceeds max_pages early? Less precise but faster.
                           # if len(visited) + len(queue) < max_pages: # Simple check
                           queue.append((normalized_next_url, current_depth + 1))
                           links_found_on_page += 1
                           # Limit links added per page? (Optional)
                           # if links_found_on_page > SOME_LIMIT: break

                print(f"  -> Added {links_found_on_page} valid links to queue.")

            except Exception as e:
                print(f"  -> Error finding/processing links on {normalized_url}: {e}")

        # Politeness delay
        if queue and pages_scraped_count < max_pages: # Avoid sleeping after the last scrape
            print(f"  -> Crawl delay ({CRAWL_DELAY}s)...")
            time.sleep(CRAWL_DELAY)

    print(f"--- Web Scraping/Crawling Phase Complete ---")
    print(f"Total unique pages visited: {len(visited)}")
    print(f"Total pages scraped successfully (content found): {sum(1 for c in scraped_content_collection.values() if c)}")

    return scraped_content_collection

# --- Example Usage (needs update if testing crawl) ---
if __name__ == '__main__':
    # --- Test Tavily Search ---
    print("\n--- Testing Tavily Search ---")
    tavily_query = "What is the current status of the Artemis program?"
    tavily_results = perform_tavily_search(tavily_query, max_results=3)
    if tavily_results:
        print(f"Tavily Results for '{tavily_query}':")
        # print(json.dumps(tavily_results, indent=2)) # Pretty print full results
        print(f"  Answer: {tavily_results.get('answer')}")
        print(f"  Result Count: {len(tavily_results.get('results', []))}")
        for res in tavily_results.get('results', []):
            print(f"    - {res.get('title')}: {res.get('url')}")
    else:
        print("Tavily search failed or API key missing.")

    # --- Test News API Search ---
    print("\n--- Testing News API Search ---")
    news_query = "AI regulations EU"
    news_results = get_news_articles(news_query, num_results=3, page=1)
    if news_results:
        print(f"NewsAPI Results for '{news_query}':")
        # print(json.dumps(news_results, indent=2))
        print(f"  Total Results Found by API: {news_results.get('totalResults')}")
        print(f"  Articles Returned (Page 1): {len(news_results.get('articles', []))}")
        for article in news_results.get('articles', []):
            print(f"    - {article.get('title')}: {article.get('url')}")
    else:
        print("NewsAPI search failed or API key missing.")

    # --- Test Scrape/Crawl (Existing) ---
    # print("\n--- Testing Scrape/Crawl (Original Test) ---")
    # search_query = "MIT CSAIL research areas"
    # search_results_serp = perform_search(search_query, num_results=2)
    # if search_results_serp:
    #      # ... (existing scrape/crawl tests using search_results_serp remain the same)
    #      pass # Placeholder to keep the structure
    # else:
    #     print("SerpApi Search failed, cannot test scraping/crawling.") 