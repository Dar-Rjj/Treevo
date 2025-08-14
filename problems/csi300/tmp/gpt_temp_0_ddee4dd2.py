import pandas as pd
import pandas as pd

def heuristics_v2(df):
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Compute short-term (5 days) and medium-term (20 days) moving averages
    df['ma_5'] = df['close'].rolling(window=5).mean()
    df['ma_20'] = df['close'].rolling(window=20).mean()
    
    # Trend signal: short-term MA minus medium-term MA
    df['trend_signal'] = df['ma_5'] - df['ma_20']
    
    # Compute True Range and Average True Range (ATR) for volatility
    df['high_low_range'] = df['high'] - df['low']
    df['true_range'] = df[['high' - 'low', (df['high'] - df['close'].shift(1)).abs(), (df['low'] - df['close'].shift(1)).abs()]].max(axis=1)
    df['atr_14'] = df['true_range'].rolling(window=14).mean()
    
    # Measure volume changes
    df['volume_change'] = df['volume'].pct_change()
    
    # Create on-balance volume (OBV) indicator
    df['obv'] = (df['close'] - df['close'].shift(1)).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0)) * df['volume']
    df['obv'] = df['obv'].cumsum()
    
    # Price-volume momentum factor
    df['price_volume_momentum'] = df['daily_return'] * df['volume_change']
    
    # Combine all factors into a single alpha factor
    df['alpha_factor'] = df['trend_signal'] + df['atr_14'] + df['obv'] + df['price_volume_momentum']
    
    return df['alpha_factor']
