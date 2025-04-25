# backend/app/main.py
import json
import asyncio
from fastapi import FastAPI, HTTPException, Request, Depends
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

# --- NEW: Import CORS Middleware --- 
from fastapi.middleware.cors import CORSMiddleware
# --- END NEW --- 

# --- NEW: Define allowed origins --- 
# You should restrict this more in a real production scenario
# For now, allow your Netlify deploy and localhost for testing
origins = [
    "https://webreasearchagent.netlify.app", # Your deployed frontend
    "http://localhost:3000",             # Your local frontend dev server
    # Add any other origins if needed
]
# --- END NEW --- 

# --- NEW: Add CORS Middleware --- 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # List of allowed origins
    allow_credentials=True,
    allow_methods=["*"],    # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],    # Allow all headers
)
# --- END NEW --- 

# --- Pydantic Model for Request Body ---
class ResearchRequest(BaseModel):
    query: str

# --- Agent Initialization (LAZY LOADING) --- 
agent_instance = None # Initialize as None globally
def get_agent():
    """FastAPI dependency to lazily load the agent."""
    global agent_instance
    print("--- ENTERING get_agent --- ")
    if agent_instance is None:
        print("Initializing WebResearchAgent instance (agent_instance is None)...")
        try:
            print("  Attempting WebResearchAgent() creation...")
            agent_instance = WebResearchAgent()
            print("  WebResearchAgent() creation apparently successful.")
            if not agent_instance.llm_client:
                 print("  Warning: Agent LLM client failed to initialize after creation.")
            else:
                 print("  Agent LLM client seems OK after creation.")
        except Exception as e:
            print(f"  FATAL: Exception during WebResearchAgent() creation: {e}")
            import traceback 
            traceback.print_exc() 
            raise HTTPException(status_code=503, detail=f"Research agent failed to initialize: {e}") 
        print("Finished WebResearchAgent initialization block.")
    else:
        print("--- Using existing agent_instance --- ")
        
    # Check after initialization attempt
    if agent_instance is None:
        print("--- Agent instance check FAILED (still None) after initialization attempt --- ")
        raise HTTPException(status_code=503, detail="Agent instance is None after initialization.")
    elif not hasattr(agent_instance, 'run_research_pipeline'):
        print("--- Agent instance check FAILED (missing method) after initialization attempt --- ")
        raise HTTPException(status_code=503, detail="Agent instance missing critical method after initialization.")
    else:
        print("--- Agent instance check PASSED after attempt --- ")
        
    print("--- EXITING get_agent (returning agent instance) --- ")
    return agent_instance
# --- END Agent Initialization --- 

# --- SSE Event Generator --- 
async def research_event_generator(query: str, agent: WebResearchAgent):
    """Runs the research pipeline step-by-step and yields SSE events.
       Accepts agent instance via dependency injection.
    """
    current_step = 0
    total_steps = 5
    results = {}
    DEFAULT_MAX_PAGES_NO_CRAWL = 3
    DEFAULT_MAX_PAGES_WITH_CRAWL = 7
    CRAWL_DEPTH_IF_ENABLED = 1
    crawl_depth_to_use = 0
    max_pages_to_use = DEFAULT_MAX_PAGES_NO_CRAWL

    try:
        # Step 1: Analyze Query
        current_step += 1
        status_key = "analyzing_query"
        yield json.dumps({"status": status_key, "progress": current_step/total_steps, "message": status_key})
        # Use await asyncio.to_thread for blocking agent calls
        analysis_result = await asyncio.to_thread(agent.analyze_query, query)
        results["query_analysis"] = analysis_result
        yield json.dumps({"status": f"{status_key}_done", "progress": current_step/total_steps, "message": "Query analysis complete.", "data": {"query_analysis": analysis_result}})
        
        if not analysis_result or analysis_result.get("intent") == "Analysis failed" or not analysis_result.get("sub_queries"):
            raise ValueError("Query analysis failed or produced no sub-queries.")

        # Step 2: Perform Web Search
        current_step += 1
        status_key = "searching_web"
        yield json.dumps({"status": status_key, "progress": current_step/total_steps, "message": status_key})
        TOTAL_RESULTS_TARGET = 10
        RESULTS_PER_SEARCH_PAGE = 5
        MAX_SEARCH_PAGES = 3
        search_results_collection = await asyncio.to_thread(
            agent.run_web_search,
            analysis_result,
            total_results_target=TOTAL_RESULTS_TARGET,
            max_pages_per_query=MAX_SEARCH_PAGES,
            results_per_page=RESULTS_PER_SEARCH_PAGE
        )
        results["search_results"] = search_results_collection
        search_summary = {q: {'api': data.get('api_used', []), 'count': len(data.get('results', []))} for q, data in search_results_collection.items()}
        yield json.dumps({"status": f"{status_key}_done", "progress": current_step/total_steps, "message": "Web search complete.", "data": {"search_results_summary": search_summary}})
        
        if not search_results_collection or all(not data.get('results') for data in search_results_collection.values()):
             raise ValueError("Web search failed to find any results for any sub-query.")
             
        # Step 3: Scrape/Crawl Web Pages
        current_step += 1
        status_key = "scraping_pages" # Match frontend
        yield json.dumps({"status": "scraping_web", "progress": current_step/total_steps, "message": status_key})
        scraped_content_map = await asyncio.to_thread(
            agent.scrape_search_results,
            search_results_collection,
            max_pages=max_pages_to_use,
            crawl_depth=crawl_depth_to_use,
            stay_on_domain=True
        )
        results["scraped_content"] = scraped_content_map
        scraping_summary = {
             "urls_processed": len(scraped_content_map),
             "successful_scrapes": sum(1 for k, v in scraped_content_map.items() if v and isinstance(v, dict) and not k.startswith(('tavily_answer::', 'newsapi_snippet::', 'tavily_snippet::', 'serpapi_snippet::'))),
             "direct_content_items": sum(1 for k in scraped_content_map if k.startswith(('tavily_answer::', 'newsapi_snippet::', 'tavily_snippet::', 'serpapi_snippet::')))
        }
        yield json.dumps({"status": "scraping_web_done", "progress": current_step/total_steps, "message": f"Scraping/Crawling complete. Items processed: {scraping_summary['urls_processed']}", "data": {"scraping_summary": scraping_summary}})
        
        # --- Step 4: Analyze Content (Corrected Logic) ---
        current_step += 1
        status_key = "analyzing_content"
        yield json.dumps({"status": status_key, "progress": current_step/total_steps, "message": status_key})
        
        content_analysis_results = {} # Initialize dict to store analysis results per item
        items_analyzed_count = 0
        content_to_analyze_map = results.get("scraped_content", {})
        original_query_for_analysis = analysis_result.get("original_query", query)
        target_data_for_analysis = analysis_result.get("target_data_points", [])
        is_news_for_analysis = analysis_result.get("is_news_focused", False)

        print(f"Starting analysis for {len(content_to_analyze_map)} scraped items...")
        # Loop through each scraped item
        for item_key, item_content in content_to_analyze_map.items():
            analysis_result_item = None
            # Basic check: Analyze only if there's content
            if item_content:
                 # Check for Tavily answer (handled directly, no LLM call needed here)
                 if item_key.startswith('tavily_answer::') and isinstance(item_content, str):
                     analysis_result_item = {"is_relevant": True, "summary": item_content, "extracted_data": None, "source_type": "tavily_answer"}
                     print(f"  -> Using direct Tavily answer for key: {item_key}")
                     items_analyzed_count += 1
                 # Otherwise, analyze using LLM (for dicts from scrape or string snippets)
                 elif isinstance(item_content, (dict, str)):
                     print(f"  -> Analyzing item: {item_key[:80]}...")
                     # Run the potentially blocking analyze_content in a thread
                     analysis_result_item = await asyncio.to_thread(
                         agent.analyze_content,
                         item_content, # Pass individual item content
                         original_query_for_analysis,
                         target_data_for_analysis,
                         is_news_for_analysis
                     )
                     items_analyzed_count += 1
                 else:
                     print(f"  -> Skipping item with unexpected content type: {item_key}")
                     analysis_result_item = {"is_relevant": False, "summary": None, "extracted_data": None, "error": "Invalid content type", "source_type": "unknown"}
            else:
                 # Handle cases where scraping failed (value is None) or item is empty
                 print(f"  -> Skipping item with no content: {item_key}")
                 analysis_result_item = {"is_relevant": False, "summary": None, "extracted_data": None, "error": "No content scraped", "source_type": "scraped_page_failed"}
                 
            # Store the result for this item
            content_analysis_results[item_key] = analysis_result_item

        results["content_analysis"] = content_analysis_results # Store the dictionary of results
        # Yield status update after processing all items
        yield json.dumps({"status": f"{status_key}_done", "progress": current_step/total_steps, "message": f"Content analysis complete (Items Processed: {items_analyzed_count}).", "data": {"content_analysis_summary": f"{items_analyzed_count} items analyzed"}}) # Maybe don't send full results here

        # Step 5: Synthesize Report
        current_step += 1
        status_key = "synthesizing_report"
        yield json.dumps({"status": status_key, "progress": current_step/total_steps, "message": status_key})
        # Call synthesize_results and expect a dictionary
        synthesis_result_dict = await asyncio.to_thread(
            agent.synthesize_results, 
            original_query_for_analysis, # Use the original query from analysis step
            content_analysis_results
        )
        # Store the report text separately if needed, though it's in the dict
        results["synthesized_report_text"] = synthesis_result_dict.get("report", "Error: Report not found in synthesis result.")
        results["relevant_sources"] = synthesis_result_dict.get("sources", [])
        
        # Yield the final event including both report and sources
        yield json.dumps({
            "status": f"{status_key}_done", 
            "progress": 1.0, # Ensure progress hits 1.0
            "message": "Report synthesis complete.", 
            "data": {
                "synthesized_report": synthesis_result_dict.get("report"), 
                "sources": synthesis_result_dict.get("sources", [])
            }
        })

    except Exception as e:
        print(f"Error during SSE generation for query \"{query}\": {e}")
        import traceback
        traceback.print_exc()
        yield json.dumps({"status": "error", "message": f"An internal server error occurred during processing: {e}"}) 

# --- SSE Endpoint --- 
@app.get("/api/research-stream")
async def research_stream_endpoint(query: str, agent: WebResearchAgent = Depends(get_agent)):
    """API endpoint to stream research results using dependency injection for agent."""
    if not query:
        async def error_gen(): 
             yield json.dumps({"status": "error", "message": "Query parameter is missing."})
        return EventSourceResponse(error_gen())
    
    print(f"Received streaming research request for query: \"{query}\"")
    return EventSourceResponse(research_event_generator(query, agent))

# --- Health Check Endpoint --- 
@app.get("/api/health")
async def health_check():
    """Simple health check endpoint. Doesn't load the agent."""
    # We don't initialize the agent here to keep health check fast
    # Could add a check to see if keys exist if needed
    return {"status": "ok"}

# Removed old /api/research endpoint as it used global agent
# Removed __main__ block as server is run by Render

# --- Run Server (for testing) ---
if __name__ == "__main__":
    print("Starting FastAPI server using Uvicorn...")
    # Run directly for simple testing, typically you'd run from terminal
    # Use 0.0.0.0 to make it accessible on your network if needed, otherwise 127.0.0.1
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True) 