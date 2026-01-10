from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
# REMOVE: from phi.tools.exa import ExaTools (This was causing the crash)
import os
import tools # Imports our custom tools

# --- CONFIGURATION ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AGENT FACTORY ---
def get_scoped_agent(ticker: str, market: str, context: str = ""):
    instructions = [
        f"You are a specialized Quant Analyst for {ticker} listed on {market}.",
        "STRICT SCOPE: You must ONLY answer questions about this specific stock.",
        "Use the 'agent_search_tool' to find live news if the context is insufficient.",
        "You have access to a 'Deep Analysis Report' provided in the context.",
        "Use the 'Deep Analysis Report' to explain past dips and spikes.",
        f"CONTEXT FROM SYSTEM: {context}" 
    ]
    
    return Agent(
        model=Groq(id="llama3-70b-8192", api_key=GROQ_API_KEY),
        tools=[
            YFinanceTools(stock_price=True, company_info=True),
            tools.agent_search_tool # <--- WE INJECT OUR CUSTOM FUNCTION HERE
        ], 
        instructions=instructions,
        show_tool_calls=True,
        markdown=True,
    )

# --- DATA MODELS ---
class Scope(BaseModel):
    ticker: str
    market: str

class ChatRequest(BaseModel):
    ticker: str
    market: str
    user_question: str
    previous_analysis_context: str 

# --- ENDPOINTS ---
@app.get("/")
def home():
    return {"status": "Butterfly Brain is Active"}

@app.post("/deep_analysis")
def run_deep_analysis(scope: Scope):
    try:
        metrics = tools.get_expert_metrics(scope.ticker)
        history_events = tools.perform_deep_scan(scope.ticker)
        
        context_report = f"""
        --- DEEP ANALYSIS REPORT FOR {scope.ticker} ({scope.market}) ---
        [EXPERT METRICS]
        - Annualized Volatility: {metrics.get('annualized_volatility_percent')}%
        - Est. Put/Call Ratio: {metrics.get('put_call_ratio_estimate')}
        - 52-Week Range: {metrics.get('52_week_low')} - {metrics.get('52_week_high')}
        
        [HISTORICAL TIMELINE & CAUSALITY]
        """
        for event in history_events:
            context_report += f"\n- {event['date']}: {event['type']} of {event['magnitude']}. Cause: {event['possible_cause']}"
            
        return {
            "status": "success",
            "report_summary": context_report,
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_with_analyst(request: ChatRequest):
    try:
        agent = get_scoped_agent(
            ticker=request.ticker, 
            market=request.market, 
            context=request.previous_analysis_context
        )
        response = agent.run(f"{request.user_question}. (Recall the Deep Analysis Report provided in instructions)")
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
