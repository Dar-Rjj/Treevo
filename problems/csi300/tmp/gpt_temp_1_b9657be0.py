import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Compute N-day Close Price Percent Change (N = 5)
    df['close_pct_change_5'] = df['close'].pct_change(5)
    
    # Apply Exponential Decay Factor to N-day Momentum
    def exponential_decay_factor(series, decay_rate=0.9):
        return series.ewm(span=decay_rate).mean()
    
    df['momentum_5_decay'] = exponential_decay_factor(df['close_pct_change_5'])
    
    # Calculate Volume-Adjusted Intraday Volatility
    df['intraday_volatility'] = (df['high'] - df['low']) / df['close']
    df['volume_adjusted_intraday_volatility'] = df['intraday_volatility'] / np.log(df['volume'] + 1)
    
    # Combine Advanced Momentum, Intraday Volatility, and High-Low Range
    df['combined_factor'] = df['momentum_5_decay'] * df['volume_adjusted_intraday_volatility'] * (df['high'] - df['low'])
    
    # Confirm with Volume Delta
    df['volume_delta'] = df['volume'] - df['volume'].shift(45)
    df['combined_factor'] = df['combined_factor'] * abs(df['volume_delta'])
    
    # Integrate with Trend Continuity and Filter
    df['trend_continuity'] = np.sign(df['close'] - df['close'].shift(1))
    df['combined_factor'] = df['combined_factor'] * df['trend_continuity']
    df['combined_factor'] = df['combined_factor'].apply(lambda x: x if abs(x) >= 0.8 else 0)
    
    # Incorporate Additional Trend Indicators
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = df['ema_12'] - df['ema_26']
    
    # Adjust Alpha Factor
    df['alpha_factor'] = df['combined_factor'] * df['macd']
    df['alpha_factor'] = df['alpha_factor'].apply(lambda x: x if abs(x) >= 0.5 else 0)
    
    return df['alpha_factor']
