import yfinance as yf
import pandas as pd
import numpy as np
import os
from exa_py import Exa

# --- INITIALIZE EXA ---
# Ensure EXA_API_KEY is set in your Render Environment Variables
try:
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

# --- 2. The Deep Analysis Tool (Powered by Exa.ai) ---
def perform_deep_scan(symbol: str):
    try:
        if not exa:
            return [{"date": "Error", "type": "Config", "possible_cause": "EXA_API_KEY missing in Render"}]

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
            
            # --- EXA NEURAL SEARCH ---
            try:
                # We ask Exa specifically for the "reason" or "news"
                query = f"Why did {symbol} stock {move_type} on {date_str}? Financial news analysis."
                
                # Exa "search_and_contents" gets us the text without clicking links
                result = exa.search_and_contents(
                    query,
                    type="neural",       # "Neural" understands meaning, not just keywords
                    use_autoprompt=True, # Exa optimizes your query automatically
                    num_results=1,
                    text=True            # We want the actual article text
                )
                
                if result.results:
                    # We take the first 300 chars of the article as the snippet
                    raw_text = result.results[0].text
                    news_snippet = raw_text[:300] + "..." if raw_text else "Content unavailable."
                else:
                    news_snippet = "No clear financial news found for this specific date."
                    
            except Exception as e:
                print(f"Exa Search Error: {e}")
                news_snippet = "Search failed (Check Logs)."
            # ---------------------------

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
