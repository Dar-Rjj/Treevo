import pandas as pd

def heuristics_v2(df):
    # Calculate the On-Balance Volume (OBV)
    df['obv'] = (df['close'].diff() > 0).astype(int) * df['volume'] - (df['close'].diff() < 0).astype(int) * df['volume']
    obv = df['obv'].cumsum()
    
    # Calculate the Commodity Channel Index (CCI) for a 14-day period
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    mean_typical = typical_price.rolling(window=14).mean()
    mad_typical = typical_price.rolling(window=14).apply(lambda x: ((x - x.mean()).abs().mean()), raw=True)
    cci = (typical_price - mean_typical) / (0.015 * mad_typical)
    
    # Combine OBV and CCI into a single heuristic measure with weights
    heuristics_matrix = (0.6 * obv + 0.4 * cci)
    
    return heuristics_matrix
