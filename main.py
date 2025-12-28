from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
import tools # Import our custom tools
from fastapi.middleware.cors import CORSMiddleware

# --- CONFIGURATION ---
# Get your Free API Key from: https://console.groq.com/keys
# Set it in Render Environment Variables as GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY") 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (v0, localhost, etc.)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
# --- DEFINE THE AGENT ---
# We create a function to get a fresh agent with specific scope
def get_scoped_agent(ticker: str, market: str, context: str = ""):
    instructions = [
        f"You are a specialized Quant Analyst for {ticker} listed on {market}.",
        "STRICT SCOPE: You must ONLY answer questions about this specific stock.",
        "Refuse to answer questions about other assets, politics, or general chit-chat.",
        "You have access to a 'Deep Analysis Report' provided in the context.",
        "Use the 'Deep Analysis Report' to explain past dips and spikes.",
        "You are a Mathematical Expert. If asked, calculate technicals.",
        "If the user asks for Put/Call ratios or Greeks, check the available metrics.",
        f"CONTEXT FROM SYSTEM: {context}" 
    ]
    
    return Agent(
        model=Groq(id="llama3-70b-8192", api_key=GROQ_API_KEY),
        tools=[YFinanceTools(stock_price=True, company_info=True), DuckDuckGo()],
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
    # The webapp sends back the analysis summary so the agent "remembers" it
    previous_analysis_context: str 

# --- ENDPOINTS ---

@app.get("/")
def home():
    return {"status": "Butterfly Brain is Active"}

@app.post("/deep_analysis")
def run_deep_analysis(scope: Scope):
    """
    Called when user clicks 'Deep Analysis'. 
    Generates the Heavy Report to be stored on the Client Side.
    """
    try:
        # 1. Get Expert Math
        metrics = tools.get_expert_metrics(scope.ticker)
        
        # 2. Get Historical Anomalies
        history_events = tools.perform_deep_scan(scope.ticker)
        
        # 3. Construct the "Context String"
        # This string will be returned to v0, and v0 will send it back during chat.
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
            "report_summary": context_report, # Save this in v0 React State!
            "metrics": metrics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
def chat_with_analyst(request: ChatRequest):
    """
    Called when user asks a question.
    """
    try:
        # 1. Spin up the agent with the strict scope and the report context
        agent = get_scoped_agent(
            ticker=request.ticker, 
            market=request.market, 
            context=request.previous_analysis_context
        )
        
        # 2. Run the query
        # We append "Use the provided context" to ensure it looks at the report
        response = agent.run(f"{request.user_question}. (Recall the Deep Analysis Report provided in instructions)")
        
        return {"response": response.content}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
