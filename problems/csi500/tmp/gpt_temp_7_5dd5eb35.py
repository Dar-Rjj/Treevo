import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate a novel and interpretable alpha factor based on the provided DataFrame.
    
    Parameters:
    - df: pandas DataFrame with columns (open, high, low, close, amount, volume) and index (date).
    
    Returns:
    - alpha_factor: pandas Series with the calculated alpha factor values indexed by (date).
    """
    # Calculate Price Momentum
    price_momentum = df['close'].pct_change(periods=12)
    
    # Calculate Volume Momentum
    volume_momentum = df['volume'].pct_change(periods=12)
    
    # Calculate Simple Moving Averages
    sma_50 = df['close'].rolling(window=50).mean()
    sma_200 = df['close'].rolling(window=200).mean()
    
    # Calculate Exponential Moving Averages
    ema_50 = df['close'].ewm(span=50, adjust=False).mean()
    ema_200 = df['close'].ewm(span=200, adjust=False).mean()
    
    # Trend Following: Compare shorter-term SMA or EMA with longer-term
    trend_sma = sma_50 - sma_200
    trend_ema = ema_50 - ema_200
    
    # High/Low Breakout
    lookback_period = 20
    high_breakout = df['close'] > df['high'].rolling(window=lookback_period).max()
    low_breakout = df['close'] < df['low'].rolling(window=lookback_period).min()
    
    # Average True Range (ATR)
    tr = df[['high', 'low', 'close']].join(df[['high', 'low', 'close']].shift(1)).dropna()
    tr['tr1'] = abs(tr['high'] - tr['low'])
    tr['tr2'] = abs(tr['high'] - tr['close'].shift(1))
    tr['tr3'] = abs(tr['low'] - tr['close'].shift(1))
    tr['true_range'] = tr[['tr1', 'tr2', 'tr3']].max(axis=1)
    atr = tr['true_range'].rolling(window=14).mean()
    
    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # On-Balance Volume (OBV)
    obv = (np.sign(df['close'].diff()) * df['volume']).cumsum()
    
    # Accumulation/Distribution Line
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfvol = mfm * df['volume']
    adl = mfvol.cumsum()
    
    # Combine all factors into a single alpha factor
    alpha_factor = (price_momentum + volume_momentum + trend_sma + trend_ema + 
                    high_breakout.astype(int) - low_breakout.astype(int) + 
                    atr + rsi + obv + adl)
    
    return alpha_factor

# Example usage:
# df = pd.read_csv('path_to_your_data.csv', parse_dates=['date'], index_col='date')
# alpha_factor = heuristics_v2(df)
# print(alpha_factor)
