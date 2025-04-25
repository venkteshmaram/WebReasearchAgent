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
    if agent_instance is None:
        print("Initializing WebResearchAgent instance...")
        try:
            agent_instance = WebResearchAgent()
            if not agent_instance.llm_client:
                 print("Warning: Agent LLM client failed to initialize.")
                 # Optional: raise HTTPException if LLM is absolutely critical
        except Exception as e:
            print(f"FATAL: Failed to initialize WebResearchAgent: {e}")
            # Raise an exception to prevent using a broken agent
            raise HTTPException(status_code=503, detail=f"Research agent failed to initialize: {e}") 
        print("WebResearchAgent instance initialized.")
    # Add a check here in case initialization failed but didn't raise
    if agent_instance is None or not hasattr(agent_instance, 'run_research_pipeline'):
         raise HTTPException(status_code=503, detail="Research agent unavailable after initialization attempt.")
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
        
        # Step 4: Analyze Content 
        current_step += 1
        status_key = "analyzing_content"
        yield json.dumps({"status": status_key, "progress": current_step/total_steps, "message": status_key})
        # Assuming analyze_content_parallel exists and handles async/threading correctly
        # If analyze_content is purely CPU-bound, asyncio.to_thread is appropriate
        # If it involves I/O, agent.py might need async modifications
        content_analysis_results = await asyncio.to_thread(
            agent.analyze_content,
            scraped_content_map,
            analysis_result.get("original_query", query),
            analysis_result.get("target_data_points", []),
            analysis_result.get("is_news_focused", False)
        )
        results["content_analysis"] = content_analysis_results
        items_analyzed_count = len(content_analysis_results)
        yield json.dumps({"status": f"{status_key}_done", "progress": current_step/total_steps, "message": f"Content analysis complete (Items Processed: {items_analyzed_count}).", "data": {"content_analysis": content_analysis_results}})

        # Step 5: Synthesize Report
        current_step += 1
        status_key = "synthesizing_report"
        yield json.dumps({"status": status_key, "progress": current_step/total_steps, "message": status_key})
        final_report = await asyncio.to_thread(agent.synthesize_results, query, content_analysis_results)
        results["synthesized_report"] = final_report
        yield json.dumps({"status": f"{status_key}_done", "progress": current_step/total_steps, "message": "Report synthesis complete.", "data": {"synthesized_report": final_report}})

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