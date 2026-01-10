import logging
import sys
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from phi.agent import Agent
from phi.model.groq import Groq

# --- 1. SETUP LOGGING (So we can see errors in Render) ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- 2. DEBUG ENVIRONMENT (Check Keys) ---
GROQ_KEY = os.getenv("GROQ_API_KEY")
EXA_KEY = os.getenv("EXA_API_KEY")

logger.info("--- STARTUP DIAGNOSTICS ---")
logger.info(f"GROQ_API_KEY Detected: {'YES' if GROQ_KEY else 'NO - CRITICAL FAILURE'}")
logger.info(f"EXA_API_KEY Detected: {'YES' if EXA_KEY else 'NO'}")
logger.info("---------------------------")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. THE SAFE AGENT (No Tools Initially) ---
def get_safe_agent(context: str = ""):
    """
    A bare-bones agent to test connection.
    If this works, the issue is the Tools.
    If this fails, the issue is Groq/Render.
    """
    try:
        return Agent(
            model=Groq(id="llama3-70b-8192", api_key=GROQ_KEY),
            # TOOLS COMMENTED OUT FOR DEBUGGING
            # tools=[...], 
            instructions=[
                "You are a helpful stock analyst.",
                "Currently running in DEBUG MODE.",
                f"Context: {context}"
            ],
            markdown=True,
        )
    except Exception as e:
        logger.error(f"Failed to initialize Agent: {e}")
        raise e

# --- DATA MODELS ---
class ChatRequest(BaseModel):
    ticker: str
    market: str
    user_question: str
    previous_analysis_context: str 

class Scope(BaseModel):
    ticker: str
    market: str

# --- ENDPOINTS ---
@app.get("/")
def home():
    return {"status": "Diagnostic Mode Active", "groq_connected": bool(GROQ_KEY)}

@app.post("/chat")
def chat_with_analyst(request: ChatRequest):
    logger.info(f"Received Chat Request for {request.ticker}")
    
    # Check Critical Failure before starting
    if not GROQ_KEY:
        logger.error("Attempted chat without GROQ_API_KEY")
        raise HTTPException(status_code=500, detail="Server Error: GROQ_API_KEY missing.")

    try:
        # 1. Initialize Safe Agent
        agent = get_safe_agent(context=request.previous_analysis_context)
        
        # 2. Run Query
        logger.info("Sending query to Groq...")
        response = agent.run(request.user_question)
        logger.info("Groq responded successfully.")
        
        return {"response": response.content}
    
    except Exception as e:
        # THIS IS THE MOST IMPORTANT LINE
        logger.exception("CRITICAL CHAT CRASH") 
        raise HTTPException(status_code=500, detail=f"Internal Error: {str(e)}")

@app.post("/deep_analysis")
def run_deep_analysis(scope: Scope):
    # Keep the original logic but wrap in logging
    try:
        import tools
        logger.info(f"Starting Deep Analysis for {scope.ticker}")
        
        metrics = tools.get_expert_metrics(scope.ticker)
        history_events = tools.perform_deep_scan(scope.ticker)
        
        context_report = f"Analysis for {scope.ticker} ({scope.market})"
        # (Shortened for debug, the full logic is in tools.py)
        
        return {
            "status": "success",
            "report_summary": context_report,
            "metrics": metrics
        }
    except Exception as e:
        logger.exception("DEEP ANALYSIS CRASH")
        raise HTTPException(status_code=500, detail=str(e))
