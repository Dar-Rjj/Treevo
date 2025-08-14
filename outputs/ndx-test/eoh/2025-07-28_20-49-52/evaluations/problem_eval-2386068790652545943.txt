import pandas as pd
    
    # Calculate simple moving averages
    sma_10 = df['close'].rolling(window=10).mean()
    sma_30 = df['close'].rolling(window=30).mean()

    # Calculate the difference between 10-day and 30-day SMA
    sma_diff = sma_10 - sma_30
    
    # Calculate the daily return
    daily_return = df['close'].pct_change()
    
    # Calculate the volume weighted average price (VWAP)
    vwap = ((df['close'] * df['volume']).cumsum() / df['volume'].cumsum())
    
    # Calculate the relative strength index (RSI) for the closing prices
    delta = df['close'].diff(1)
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean().apply(lambda x: max(x, 1e-5))
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Construct the heuristics matrix
    heuristics_matrix = pd.DataFrame({
        'SMA_Diff': sma_diff,
        'Daily_Return': daily_return,
        'VWAP': vwap,
        'RSI': rsi
    })
    
    # Return the heuristics matrix
    return heuristics_matrix
