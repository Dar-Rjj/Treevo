def heuristics_v2(df):
    # Calculate price momentum
    df['momentum_10'] = df['close'].pct_change(10)
    df['momentum_30'] = df['close'].pct_change(30)
    
    # Identify support and resistance levels
    df['support'] = df['low'].value_counts().idxmax()
    df['resistance'] = df['high'].value_counts().idxmax()
    
    # Analyze intraday volatility
    df['range'] = df['high'] - df['low']
    df['avg_range_5'] = df['range'].rolling(window=5).mean()
    df['std_range_20'] = df['range'].rolling(window=20).std()
    
    # Detect volume spikes
    df['avg_volume_20'] = df['volume'].rolling(window=20).mean()
    df['volume_spike'] = (df['volume'] > 2 * df['avg_volume_20']).astype(int)
    
    # Calculate the money flow index (MFI)
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['positive_money_flow'] = df['typical_price'] * df['amount'] * (df['typical_price'] > df['typical_price'].shift())
    df['negative_money_flow'] = df['typical_price'] * df['amount'] * (df['typical_price'] < df['typical_price'].shift())
    df['money_flow_ratio'] = df['positive_money_flow'].rolling(window=14).sum() / (df['positive_money_flow'].rolling(window=14).sum() + df['negative_money_flow'].rolling(window=14).sum())
    df['mfi'] = 100 - (100 / (1 + df['money_flow_ratio']))
    
    # Implement moving averages
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['sma_200'] = df['close'].rolling(window=200).mean()
    df['sma_cross'] = (df['sma_50'] > df['sma_200']).astype(int) - (df['sma_50'] < df['sma_200']).astype(int)
    
    # Use the Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    
    # Evaluate the direction of the market
    df['direction'] = (df['close'] > df['close'].shift()).astype(int)
    df['market_trend'] = df['direction'].rolling(window=14).mean()
    
    # Assess breadth indicators
    df['advancing_stocks'] = (df['close'] > df['close'].shift()).astype(int)
    df['declining_stocks'] = (df['close'] < df['close'].shift()).astype(int)
    df['advance_decline_line'] = df['advancing_stocks'].cumsum() - df['declining_stocks'].cumsum()
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (
        0.2 * df['momentum_10'] +
        0.2 * df['momentum_30'] +
        0.1 * (df['close'] - df['support']) / df['support'] +
        0.1 * (df['resistance'] - df['close']) / df['close'] +
        0.1 * df['avg_range_5'] / df['std_range_20'] +
        0.1 * df['volume_spike'] +
        0.1 * df['mfi'] +
        0.1 * df['sma_cross'] +
        0.1 * (df['rsi_overbought'] - df['rsi_oversold']) +
        0.1 * (2 * df['market_trend'] - 1) +
        0.1 * df['advance_decline_line']
    )
    
    return df['
