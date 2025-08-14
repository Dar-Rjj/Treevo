def heuristics_v2(df):
    # Define lookback periods
    sma_short = 20
    sma_long = 50
    vol_adjusted_lookback = 10
    momentum_lookback = 10
    pct_change_lookback = 5
    liquidity_lookback = 20
    dynamic_window_range = range(10, 60, 10)
    
    # Calculate Simple Moving Average (SMA) of Close Prices
    df['SMA_Short'] = df['close'].rolling(window=sma_short).mean()
    df['SMA_Long'] = df['close'].rolling(window=sma_long).mean()
    
    # Compute Volume-Adjusted Volatility
    df['High_Low_Diff'] = df['high'] - df['low']
    df['Volume_Adjusted_Volatility'] = (df['High_Low_Diff'] * df['volume']).rolling(window=vol_adjusted_lookback).mean()
    
    # Compute Price Momentum
    df['Momentum'] = (df['close'] - df['SMA_Short']) / df['close'].rolling(window=momentum_lookback).mean()
    
    # Incorporate Additional Price Change Metrics
    df['Pct_Change'] = df['close'].pct_change(periods=pct_change_lookback)
    df['High_Low_Range'] = df['high'] - df['low']
    
    # Consider Market Trend Alignment
    df['Trend_Indicator'] = (df['SMA_Short'] > df['SMA_Long']).astype(int)
    
    # Incorporate Liquidity Measures
    df['Daily_Turnover'] = df['volume'] * df['close']
    df['Liquidity'] = df['Daily_Turnover'].rolling(window=liquidity_lookback).mean()
    
    # Dynamically Adjust Component Weights Based on Market Trend and Liquidity
    initial_weights = {
        'Momentum': 0.3,
        'Volume_Adjusted_Volatility': 0.2,
        'Pct_Change': 0.2,
        'High_Low_Range': 0.1,
        'Trend_Indicator': 0.1,
        'Liquidity': 0.1
    }
    
    def adjust_weights(trend, liquidity):
        if trend == 1 and liquidity > 0:
            return {
