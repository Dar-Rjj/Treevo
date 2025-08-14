import numpy as np
def heuristics_v2(df):
    # Calculate price changes over different time frames
    df['daily_price_change'] = df['close'].diff()
    df['weekly_price_change'] = df['close'] - df['close'].shift(5)
    df['monthly_price_change'] = df['close'] - df['close'].shift(20)
    
    # Calculate moving averages
    df['SMA_5'] = df['close'].rolling(window=5).mean()
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['SMA_30'] = df['close'].rolling(window=30).mean()
    
    # Calculate differences between moving averages
    df['SMA_5_10_diff'] = df['SMA_5'] - df['SMA_10']
    df['SMA_10_30_diff'] = df['SMA_10'] - df['SMA_30']
    
    # Calculate volume trends
    df['daily_volume_change'] = df['volume'].diff()
    df['VMA_5'] = df['volume'].rolling(window=5).mean()
    df['VMA_10'] = df['volume'].rolling(window=10).mean()
    df['VMA_5_10_diff'] = df['VMA_5'] - df['VMA_10']
    
    # Analyze price and volume relationship
    df['price_to_volume_ratio'] = df['daily_price_change'] / (df['daily_volume_change'] + 1e-6)  # Adding a small constant to avoid division by zero
    df['MA_5_price_to_volume_ratio'] = df['price_to_volume_ratio'].rolling(window=5).mean()
    
    # Study volatility
    df['true_range'] = df[['high', 'close']].max(axis=1) - df[['low', 'close']].min(axis=1)
    df['ATR_10'] = df['true_range'].rolling(window=10).mean()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
