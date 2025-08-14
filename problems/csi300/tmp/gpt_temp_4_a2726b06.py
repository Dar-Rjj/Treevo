import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df, macroeconomic_data):
    # Calculate bid-ask spread proxy
    df['bid_ask_spread'] = df['high'] - df['low']
    
    # Adjust momentum by market depth (inverse of bid-ask spread)
    df['momentum'] = df['close'].pct_change()
    df['adjusted_momentum'] = df['momentum'] / df['bid_ask_spread']
    
    # Short-term and long-term window sizing based on historical volatility
    def determine_window_size(volatility):
        if volatility > 0.05:  # high volatility
            return 5
        else:
            return 20
    
    volatility = df['close'].rolling(window=20).std()
    short_term_window = determine_window_size(volatility.iloc[-1])
    long_term_window = short_term_window * 4
    
    # Short-term momentum with volume-weighted exponential smoothing
    short_momentum = df['close'].ewm(span=short_term_window).mean().pct_change()
    short_momentum = short_momentum * df['volume'].ewm(span=short_term_window).mean()
    
    # Long-term momentum with volume-weighted exponential smoothing
    long_momentum = df['close'].ewm(span=long_term_window).mean().pct_change()
    long_momentum = long_momentum * df['volume'].ewm(span=long_term_window).mean()
    
    # Combine short-term and long-term momentum
    combined_momentum = (short_momentum + long_momentum) / 2
    
    # Integrate macroeconomic indicators
    gdp_growth_rate = macroeconomic_data['gdp_growth_rate']
    interest_rates = macroeconomic_data['interest_rates']
    inflation = macroeconomic_data['inflation']
    
    # Define macroeconomic sentiment score
    macro_sentiment = 1 if gdp_growth_rate > 0.03 else -1
    macro_sentiment += 1 if interest_rates < 0.02 else -1
    macro_sentiment += 1 if inflation < 0.02 else -1
    
    # Weight momentum by macroeconomic sentiment score
    final_momentum = combined_momentum * macro_sentiment
    
    return final_momentum
