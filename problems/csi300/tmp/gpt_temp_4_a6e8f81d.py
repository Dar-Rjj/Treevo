def heuristics_v2(df):
    high, low, close, volume, amount = df['high'], df['low'], df['close'], df['volume'], df['amount']
    
    volatility = (high - low) / close
    price_change = close.pct_change()
    volume_ma = volume.rolling(window=10).mean()
    
    momentum = price_change.rolling(window=5).mean()
    volatility_adjusted_momentum = momentum / volatility.rolling(window=5).mean()
    
    reversal_signal = (close - close.rolling(window=5).mean()) / close.rolling(window=5).std()
    volume_confirmation = (volume > volume_ma).astype(int)
    
    heuristics_matrix = volatility_adjusted_momentum * reversal_signal * volume_confirmation
    return heuristics_matrix
