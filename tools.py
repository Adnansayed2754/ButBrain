import yfinance as yf
import pandas as pd
import numpy as np
import os
from exa_py import Exa

# --- INITIALIZE EXA ---
try:
    # We initialize it once here to use everywhere
    exa = Exa(api_key=os.getenv("EXA_API_KEY"))
except Exception as e:
    print(f"Exa Initialization Warning: {e}")
    exa = None

# --- 1. The Expert Math Tools ---
def get_expert_metrics(symbol: str):
    """Calculates advanced options & volatility metrics."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        
        if hist.empty:
            return {"error": "No price data found."}
        
        # Volatility
        hist['returns'] = hist['Close'].pct_change()
        volatility = hist['returns'].std() * np.sqrt(252) * 100
        
        # PCR Fallback
        pcr = "Data Unavailable"
        try:
            exps = ticker.options
            if exps:
                opt = ticker.option_chain(exps[0])
                puts_vol = opt.puts['volume'].sum()
                calls_vol = opt.calls['volume'].sum()
                if calls_vol > 0:
                    pcr = round(puts_vol / calls_vol, 2)
        except:
            pass

        return {
            "annualized_volatility_percent": round(volatility, 2),
            "put_call_ratio_estimate": pcr,
            "current_price": round(hist['Close'].iloc[-1], 2),
            "52_week_high": round(hist['High'].max(), 2),
            "52_week_low": round(hist['Low'].min(), 2)
        }
    except Exception as e:
        print(f"Metrics Error: {e}")
        return {"error": str(e)}

# --- 2. The Deep Analysis Tool ---
def perform_deep_scan(symbol: str):
    """Scans 2 years of data for anomalies and uses Exa to find reasons."""
    try:
        if not exa:
            return [{"date": "Error", "type": "Config", "possible_cause": "EXA_API_KEY missing"}]

        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="2y")
        
        if hist.empty:
            return []

        # Detect Anomalies
        hist['Pct_Change'] = hist['Close'].pct_change() * 100
        anomalies = hist[ (hist['Pct_Change'] > 5) | (hist['Pct_Change'] < -5) ]
        
        events_log = []
        top_moves = anomalies.reindex(anomalies['Pct_Change'].abs().sort_values(ascending=False).index).head(3)
        
        for date, row in top_moves.iterrows():
            move_type = "SPIKE" if row['Pct_Change'] > 0 else "CRASH"
            date_str = date.strftime('%Y-%m-%d')
            
            # --- FIXED EXA SEARCH ---
            try:
                query = f"Why did {symbol} stock {move_type} on {date_str}? Financial news."
                
                # REMOVED 'use_autoprompt' to fix the error
                result = exa.search_and_contents(
                    query,
                    type="neural",
                    num_results=1,
                    text=True
                )
                
                if result.results:
                    raw_text = result.results[0].text
                    news_snippet = raw_text[:300] + "..." if raw_text else "Content unavailable."
                else:
                    news_snippet = "No clear financial news found."
                    
            except Exception as e:
                print(f"Exa Search Error for {date_str}: {e}")
                news_snippet = "Search failed."
            # ------------------------

            events_log.append({
                "date": date_str,
                "type": move_type,
                "magnitude": f"{row['Pct_Change']:.2f}%",
                "possible_cause": news_snippet
            })
            
        return events_log
        
    except Exception as e:
        print(f"Deep Scan Error: {e}")
        return []

# --- 3. NEW: Custom Search Tool for the Agent ---
def agent_search_tool(query: str) -> str:
    """
    Use this tool to search the web for live financial news and information.
    Args:
        query (str): The question or topic to search for.
    """
    try:
        if not exa:
            return "Search unavailable (API Key missing)."
            
        # The Agent uses this simple function instead of the broken ExaTools wrapper
        result = exa.search_and_contents(
            query,
            type="neural",
            num_results=3,
            text=True
        )
        
        # Format results nicely for the Agent to read
        response = ""
        for item in result.results:
            response += f"--- Source: {item.title} ---\n{item.text[:500]}...\n\n"
            
        return response if response else "No results found."
    except Exception as e:
        return f"Search failed: {str(e)}"
