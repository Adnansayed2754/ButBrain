import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from duckduckgo_search import DDGS

# --- 1. The Expert Math Tools ---
def get_expert_metrics(symbol: str):
    """Calculates advanced options & volatility metrics."""
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="1y")
        
        # A. Volatility (Annualized)
        hist['returns'] = hist['Close'].pct_change()
        volatility = hist['returns'].std() * np.sqrt(252) * 100
        
        # B. Put/Call Ratio (Proxy using Volume if real options data fails)
        # Note: Real-time PCR is expensive. We approximate using recent options chain if available.
        pcr = "Data Unavailable (Requires Paid API)"
        try:
            # Attempt to fetch nearest expiry options
            exps = ticker.options
            if exps:
                opt = ticker.option_chain(exps[0])
                puts_vol = opt.puts['volume'].sum()
                calls_vol = opt.calls['volume'].sum()
                if calls_vol > 0:
                    pcr = round(puts_vol / calls_vol, 2)
        except:
            pass # Fallback if options data is unparsable

        return {
            "annualized_volatility_percent": round(volatility, 2),
            "put_call_ratio_estimate": pcr,
            "current_price": round(hist['Close'].iloc[-1], 2),
            "52_week_high": round(hist['High'].max(), 2),
            "52_week_low": round(hist['Low'].min(), 2)
        }
    except Exception as e:
        return {"error": str(e)}

# --- 2. The Deep Analysis Tool ---
def perform_deep_scan(symbol: str):
    """Scans 2 years of data for anomalies and matches news."""
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="2y")
    
    # Detect Anomalies (Price drop > 5% or Spike > 5%)
    hist['Pct_Change'] = hist['Close'].pct_change() * 100
    anomalies = hist[ (hist['Pct_Change'] > 5) | (hist['Pct_Change'] < -5) ]
    
    events_log = []
    
    # For the top 3 biggest moves, find "Context"
    # (Sorting by absolute magnitude of move)
    top_moves = anomalies.reindex(anomalies['Pct_Change'].abs().sort_values(ascending=False).index).head(3)
    
    for date, row in top_moves.iterrows():
        move_type = "SPIKE" if row['Pct_Change'] > 0 else "CRASH"
        date_str = date.strftime('%Y-%m-%d')
        
        # Search News for that date
        # Note: We use a generic search here since we don't have a paid History News API
        # We search "Ticker + Date + News"
        query = f"{symbol} stock news {date_str} reason"
        results = DDGS().text(query, max_results=1)
        news_snippet = results[0]['body'] if results else "No specific news found."
        
        events_log.append({
            "date": date_str,
            "type": move_type,
            "magnitude": f"{row['Pct_Change']:.2f}%",
            "possible_cause": news_snippet
        })
        
    return events_log
