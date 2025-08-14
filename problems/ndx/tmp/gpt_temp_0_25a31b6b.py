import pandas as pd

def heuristics_v2(df):
    # Earnings yield: earnings per share / close price
    earnings_yield = df['eps'] / df['close']
    
    # Price-to-Earnings Growth Ratio (PEG): P/E ratio / earnings growth rate
    pe_ratio = df['close'] / df['eps']
    peg_ratio = pe_ratio / df['earnings_growth_rate']
    
    # Rate of change in earnings per share over 4 quarters
    eps_change = df['eps'].pct_change(periods=4)
    
    # Composite heuristics matrix calculation
    heuristics_matrix = earnings_yield - peg_ratio + eps_change
    
    return heuristics_matrix
