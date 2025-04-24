# backend/app/main.py
import json
import asyncio
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from sse_starlette.sse import EventSourceResponse

# Import agent and tools needed for pipeline steps using relative imports
from .agent import WebResearchAgent
from .tools import scrape_web_page # Import the specific tool function

# Add project root to path to find agent and tools modules -> REMOVED
# project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

try:
    # No longer need try/except here as imports should work directly
    pass
except ImportError:
     # Dummy definition can be removed or kept as fallback if desired
     # print(f"Error: Could not import WebResearchAgent. Ensure agent.py is in the project root ({project_root}) or Python path.")
     # class WebResearchAgent:
     #    ...
     pass # Or raise an error if agent is critical

app = FastAPI()

# --- CORS Configuration ---
# Allows requests from your frontend development server
origins = [
    "http://localhost",      # Allow base localhost
    "http://localhost:3000", # Default React dev port
    "http://localhost:5173", # Default Vite dev port
    # Add any other origins if your frontend runs elsewhere
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods (GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

# --- Pydantic Model for Request Body ---
class ResearchRequest(BaseModel):
    query: str

# --- Agent Initialization ---
# Initialize the agent once when the API starts
# This assumes API keys are correctly set in the .env file in the project root
# The dotenv loading within agent.py and tools.py should still find .env in parent dir
try:
    agent_instance = WebResearchAgent()
    if not agent_instance.llm_client:
         print("Warning: Agent LLM client failed to initialize during API startup.")
except Exception as e:
    print(f"FATAL: Failed to initialize WebResearchAgent during API startup: {e}")
    # Provide a dummy agent if initialization fails catastrophically
    # Define the dummy class here if needed, or ensure it's defined above
    class DummyAgent:
        def __init__(self, *args, **kwargs): print("Using DummyAgent due to init failure.")
        def run_research_pipeline(self, *args, **kwargs): return {"error": "Agent initialization failed"}
    agent_instance = DummyAgent()

# --- Original API Endpoint (kept for non-streaming testing) ---
@app.post("/api/research")
async def run_research(request: ResearchRequest):
    """
    Endpoint to run the full web research pipeline (non-streaming).
    Takes a user query, analyzes it, performs web searches,
    scrapes content, analyzes content, and returns the combined results at the end.
    """
    print(f"Received non-streaming research request for query: \"{request.query}\"")
    if not agent_instance or not hasattr(agent_instance, 'run_research_pipeline'):
         raise HTTPException(status_code=500, detail="Research agent is not properly configured or failed to load.")
    try:
        final_result = agent_instance.run_research_pipeline(
            user_query=request.query,
            max_results_per_query=5,
            max_urls_to_scrape=3
        )
        print(f"Non-streaming research pipeline completed for query: \"{request.query}\". Returning results.")
        return final_result
    except Exception as e:
        print(f"Error during non-streaming research pipeline for query \"{request.query}\": {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- NEW: SSE Event Generator --- 
async def research_event_generator(query: str):
    """Runs the research pipeline step-by-step and yields SSE events."""
    current_step = 0
    total_steps = 5 # 1:Analyze, 2:Search, 3:Scrape/Crawl, 4:Content Analysis, 5:Synthesize
    results = {}
    
    # --- Define Default Limits --- 
    DEFAULT_MAX_PAGES_NO_CRAWL = 3 # Limit pages if not crawling
    DEFAULT_MAX_PAGES_WITH_CRAWL = 7 # Limit pages if crawling (depth 1)
    CRAWL_DEPTH_IF_ENABLED = 1     # Fixed crawl depth when enabled

    # Crawl parameters will be determined after query analysis
    crawl_depth_to_use = 0
    max_pages_to_use = DEFAULT_MAX_PAGES_NO_CRAWL

    if not agent_instance:
        yield json.dumps({"status": "error", "message": "Agent not initialized"})
        return

    try:
        # Step 1: Analyze Query
        current_step += 1
        status_key = "analyzing_query"
        yield json.dumps({"status": status_key, "progress": current_step/total_steps, "message": status_key}) # Send key in message
        analysis_result = await asyncio.to_thread(agent_instance.analyze_query, query)
        results["query_analysis"] = analysis_result
        yield json.dumps({"status": f"{status_key}_done", "progress": current_step/total_steps, "message": "Query analysis complete.", "data": {"query_analysis": analysis_result}})
        
        if not analysis_result or analysis_result.get("intent") == "Analysis failed" or not analysis_result.get("sub_queries"):
            yield json.dumps({"status": "error", "message": "Query analysis failed or produced no sub-queries. Stopping."})
            return
            
        # --- Step 2: Perform Web Search (with Pagination) ---
        current_step += 1
        status_key = "searching_web"
        yield json.dumps({"status": status_key, "progress": current_step/total_steps, "message": status_key}) # Send key in message

        # Define pagination parameters for the search
        TOTAL_RESULTS_TARGET = 10 # Desired total results per sub-query
        RESULTS_PER_SEARCH_PAGE = 5 # How many items to fetch per API call page
        MAX_SEARCH_PAGES = 3 # Max API pages to fetch per sub-query

        search_results_collection = await asyncio.to_thread(
            agent_instance.run_web_search,
            analysis_result,
            total_results_target=TOTAL_RESULTS_TARGET,
            max_pages_per_query=MAX_SEARCH_PAGES,
            results_per_page=RESULTS_PER_SEARCH_PAGE
        )
        results["search_results"] = search_results_collection
        search_summary = {q: {'api': data.get('api_used', 'none'), 'count': len(data.get('results', []))} for q, data in search_results_collection.items()}
        yield json.dumps({"status": f"{status_key}_done", "progress": current_step/total_steps, "message": f"Web search complete.", "data": {"search_results_summary": search_summary}})

        if not search_results_collection or all(not data.get('results') for data in search_results_collection.values()):
            yield json.dumps({"status": "error", "message": "Web search failed to find any results for any sub-query. Stopping."})
            return

        # --- Step 3: Scrape/Crawl Web Pages ---
        current_step += 1
        # Match the frontend key "scraping_pages"
        status_key = "scraping_pages" 
        yield json.dumps({"status": "scraping_web", "progress": current_step/total_steps, "message": status_key}) # Send key in message
        scraped_content_map = await asyncio.to_thread(
            agent_instance.scrape_search_results,
            search_results_collection,
            max_pages=max_pages_to_use,
            crawl_depth=crawl_depth_to_use,
            stay_on_domain=True
        )
        results["scraped_content"] = scraped_content_map
        scraping_summary = {
             "urls_processed": len(scraped_content_map),
             "successful_scrapes": sum(1 for k, v in scraped_content_map.items() if v and not k.startswith(('tavily_answer::', 'newsapi_snippet::', 'tavily_snippet::', 'serpapi_snippet::'))),
             "direct_content_items": sum(1 for k in scraped_content_map if k.startswith(('tavily_answer::', 'newsapi_snippet::', 'tavily_snippet::', 'serpapi_snippet::')))
        }
        yield json.dumps({"status": "scraping_web_done", "progress": current_step/total_steps, "message": f"Scraping/Crawling complete. Items processed: {scraping_summary['urls_processed']}", "data": {"scraping_summary": scraping_summary}})

        # --- Step 4: Analyze Content ---
        current_step += 1
        status_key = "analyzing_content"
        yield json.dumps({"status": status_key, "progress": current_step/total_steps, "message": status_key}) # Send key in message
        
        content_analysis_results = {}
        analysis_tasks = []
        item_keys_for_analysis = []

        content_to_analyze = scraped_content_map if scraped_content_map else {}
        target_data_points = analysis_result.get("target_data_points", [])
        is_news_query = analysis_result.get("is_news_focused", False)

        for item_key, text in content_to_analyze.items():
             if text and not item_key.startswith('tavily_answer::'):
                source_type = "scraped_page"
                if item_key.startswith(('newsapi_snippet::', 'tavily_snippet::', 'serpapi_snippet::')):
                    source_type = item_key.split('_snippet::')[0] + "_snippet"
                analysis_tasks.append(asyncio.to_thread(agent_instance.analyze_content, text, query, target_data_points, is_news_query))
                item_keys_for_analysis.append(item_key)

        for item_key, text in content_to_analyze.items():
             if item_key.startswith('tavily_answer::'):
                 sub_q = item_key.split('::')[1]
                 print(f"Including direct Tavily answer for sub-query: {sub_q}")
                 content_analysis_results[item_key] = {"is_relevant": True, "summary": text, "extracted_data": None, "source_type": "tavily_answer"}

        if analysis_tasks:
            print(f"Running {len(analysis_tasks)} content analysis tasks...")
            analysis_results_list = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            for i, result_or_exc in enumerate(analysis_results_list):
                 item_key = item_keys_for_analysis[i]
                 if isinstance(result_or_exc, Exception):
                      print(f"Error during parallel content analysis for {item_key}: {result_or_exc}")
                      content_analysis_results[item_key] = {"is_relevant": False, "summary": None, "extracted_data": None, "error": "Analysis task failed"}
                 else:
                      if "source_type" not in result_or_exc:
                           if item_key.startswith(('newsapi_snippet::', 'tavily_snippet::', 'serpapi_snippet::')):
                                result_or_exc["source_type"] = item_key.split('_snippet::')[0] + "_snippet"
                           else:
                                result_or_exc["source_type"] = "scraped_page"
                      content_analysis_results[item_key] = result_or_exc

        for item_key, text in content_to_analyze.items():
            if not text and item_key not in content_analysis_results:
                 content_analysis_results[item_key] = {"is_relevant": False, "summary": None, "extracted_data": None, "error": "No content scraped or analyzed"}

        results["content_analysis"] = content_analysis_results
        items_analyzed_count = len(analysis_tasks) + sum(1 for k in content_analysis_results if k.startswith('tavily_answer::'))
        yield json.dumps({"status": f"{status_key}_done", "progress": current_step/total_steps, "message": f"Content analysis complete (Items Processed: {items_analyzed_count}).", "data": {"content_analysis": content_analysis_results}})

        # --- Step 5: Synthesize Report ---
        current_step += 1
        status_key = "synthesizing_report"
        yield json.dumps({"status": status_key, "progress": current_step/total_steps, "message": status_key}) # Send key in message
        final_report = await asyncio.to_thread(agent_instance.synthesize_results, query, content_analysis_results)
        results["synthesized_report"] = final_report
        yield json.dumps({"status": f"{status_key}_done", "progress": current_step/total_steps, "message": "Report synthesis complete.", "data": {"synthesized_report": final_report}})

    except Exception as e:
        print(f"Error during SSE generation for query \"{query}\": {e}")
        import traceback
        traceback.print_exc()
        yield json.dumps({"status": "error", "message": f"An internal server error occurred during processing: {e}"}) 

# --- NEW: SSE Endpoint --- 
@app.get("/api/research-stream") # Using GET for simplicity with EventSource
async def research_stream(request: Request, query: str):
    """Endpoint to run the research pipeline and stream results via SSE."""
    print(f"Received streaming research request for query: \"{query}\"")
    if not query:
        # Normally use HTTPException for REST, but need to yield error for SSE
        async def error_gen(): 
             yield json.dumps({"status": "error", "message": "Query parameter is missing."}) 
        return EventSourceResponse(error_gen())
        
    # Call the generator function wrapped in EventSourceResponse
    return EventSourceResponse(research_event_generator(query))

# --- Health Check Endpoint ---
@app.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "agent_initialized": agent_instance is not None and agent_instance.llm_client is not None}


# --- Run Server (for testing) ---
if __name__ == "__main__":
    print("Starting FastAPI server using Uvicorn...")
    # Run directly for simple testing, typically you'd run from terminal
    # Use 0.0.0.0 to make it accessible on your network if needed, otherwise 127.0.0.1
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 