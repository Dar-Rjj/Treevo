import pandas as pd
    
    # Short and long EMA calculation
    short_ema = df['close'].ewm(span=12, adjust=False).mean()
    long_ema = df['close'].ewm(span=26, adjust=False).mean()
    
    # EMA crossover signal
    ema_crossover = short_ema - long_ema
    
    # Rate of change in volume
    roc_volume = df['volume'].pct_change(periods=10)
    
    # Combine heuristics into a DataFrame
    heuristics_matrix = pd.DataFrame({'EMA_Crossover': ema_crossover, 'ROC_Volume': roc_volume}).fillna(0)
    
    return heuristics_matrix
