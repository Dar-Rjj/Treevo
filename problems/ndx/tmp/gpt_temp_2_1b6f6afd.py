import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate daily price change
    df['price_change'] = df['close'].diff()
    
    # Calculate daily price range
    df['price_range'] = df['high'] - df['low']
    
    # Calculate daily volume change
    df['volume_change'] = df['volume'].diff()
    
    # Calculate average volume over a period (e.g., 10 days)
    df['avg_volume_10d'] = df['volume'].rolling(window=10).mean()
    
    # Calculate on-balance volume (OBV)
    df['obv'] = (df['close'].diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']).cumsum()
    
    # Calculate volume-weighted average price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Calculate simple moving average (SMA) of prices (e.g., 20 days)
    df['sma_20d'] = df['close'].rolling(window=20).mean()
    
    # Calculate exponential moving average (EMA) of prices (e.g., 20 days, smoothing factor 0.1)
    df['ema_20d'] = df['close'].ewm(span=20, adjust=False).mean()
    
    # Calculate relative strength index (RSI) (e.g., 14 days)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))
    
    # Calculate standard deviation of daily returns (e.g., 30 days)
    df['std_dev_30d'] = df['price_change'].rolling(window=30).std()
    
    # Calculate average true range (ATR) (e.g., 14 days)
    df['tr'] = df[['high' - 'low', 
                   (df['high'] - df['close'].shift()).abs(), 
                   (df['low'] - df['close'].shift()).abs()]].max(axis=1)
    df['atr_14d'] = df['tr'].rolling(window=14).mean()
    
    # Calculate intraday high-low ratio
    df['intraday_high_low_ratio'] = df['high'] / df['low']
    
    # Calculate close-high ratio
    df['close_high_ratio'] = df['close'] / df['high']
    
    # Calculate close-low ratio
    df['close_low_ratio'] = df['close'] / df['low']
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = (df['price_change'] + df['price_range'] + df['volume_change'] + df['avg_volume_10d'] + df['obv'] + df['vwap'] + df['sma_20d'] + df['ema_20d'] + df['rsi_14d'] + df['std_dev_30d'] + df['atr_14d'] + df['intraday_high_low_ratio'] + df['close_high_ratio'] + df['close_low_ratio']) / 14
    
    return df['alpha_factor']
