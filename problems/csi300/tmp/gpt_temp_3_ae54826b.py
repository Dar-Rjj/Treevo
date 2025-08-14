import pandas as pd
def heuristics_v2(df: pd.DataFrame) -> pd.Series:
    # Calculate the daily return
    df['daily_return'] = df['close'].pct_change()
    
    # Calculate the 10-day average of daily returns to capture short-term momentum
    df['momentum_10d'] = df['daily_return'].rolling(window=10).mean()
    
    # Calculate the 30-day standard deviation of daily returns to capture volatility
    df['volatility_30d'] = df['daily_return'].rolling(window=30).std()
    
    # Calculate the 5-day exponential moving average (EMA) of volume to capture liquidity
    df['volume_ema_5d'] = df['volume'].ewm(span=5, adjust=False).mean()
    
    # Calculate the 10-day average true range (ATR) for volatility
    df['tr_high_low'] = df['high'] - df['low']
    df['tr_high_close_prev'] = (df['high'] - df['close'].shift()).abs()
    df['tr_low_close_prev'] = (df['low'] - df['close'].shift()).abs()
    df['true_range'] = df[['tr_high_low', 'tr_high_close_prev', 'tr_low_close_prev']].max(axis=1)
    df['atr_10d'] = df['true_range'].rolling(window=10).mean()
    
    # Calculate the 20-day price change to capture longer-term momentum
    df['price_change_20d'] = (df['close'] / df['close'].shift(20)) - 1
    
    # Calculate the 50-day price change to capture even longer-term trends
    df['price_change_50d'] = (df['close'] / df['close'].shift(50)) - 1
    
    # Integrate a simple moving average crossover as a technical indicator
    df['sma_50d'] = df['close'].rolling(window=50).mean()
    df['sma_200d'] = df['close'].rolling(window=200).mean()
    df['sma_crossover'] = df['sma_50d'] > df['sma_200d']
    
    # Calculate the 14-day Relative Strength Index (RSI) for momentum
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))
    
    # Calculate the 20-day Moving Average Convergence Divergence (MACD)
    df['ema_12d'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26d'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd_line'] = df['ema_12d'] - df['ema_26d']
    df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
    df['macd_histogram'] = df['macd_line'] - df['signal_line']
    
    # Combine the factors into a single alpha factor
    factor = (
        df['momentum_10d'] * 
        (1 / df['volatility_30d']) * 
        df['volume_ema_5d'] * 
        df['atr_10d'] * 
        df['price_change_20d'] * 
        df['price_change_50d'] * 
        df['sma_crossover'].astype(int) * 
        df['rsi_14d'] * 
        df['macd_histogram']
    )
    
    # Apply machine learning for feature weighting
    from sklearn.decomposition import PCA
    pca = PCA(n_components=1)
    pca_factor = pca.fit_transform(factor.to_frame())
    factor = pd.Series(pca_factor.flatten(), index=factor.index)
    
    return factor
