import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Daily Price Movement Range
    df['price_range'] = df['high'] - df['low']
    
    # Calculate Volume Weighted Average Price (VWAP)
    df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
    
    # Determine Daily Return Deviation from VWAP
    df['deviation'] = df['close'] - df['vwap']
    
    # Identify Trend Reversal Potential
    df['trend_reversal'] = (df['deviation'] > df['deviation'].shift(1)) & (df['volume'] > df['volume'].rolling(window=5).mean())
    
    # Calculate Intraday Return
    df['intraday_return'] = (df['high'] - df['low']) / df['open']
    
    # Adjust for Volume
    df['vol_adjusted_return'] = (df['volume'] - df['volume'].rolling(window=7).mean()) * df['intraday_return']
    
    # Incorporate Price Volatility
    std_7d = df['close'].rolling(window=7).std()
    df['volatility_adjusted_return'] = df['intraday_return'] * (1.5 if std_7d > std_7d.mean() else 0.5)
    
    # Incorporate Momentum Shift
    df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['momentum_shift'] = np.where(df['ema_5'] > df['ema_20'], df['intraday_return'] + 0.8, df['intraday_return'] - 0.8)
    
    # Evaluate Price Efficiency
    df['price_efficiency'] = (df['high'] + df['low']) / 2 - df['vwap']
    
    # Combine Indicators
    combined_score = (
        df['trend_reversal'] * df['volatility_adjusted_return'] +
        np.where(df['price_efficiency'] > 0, 1, -1) * df['momentum_shift']
    )
    
    # Apply Final Smoothing Factor
    combined_score *= 0.9
    
    # Incorporate Historical Price Pattern
    df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
    historical_pattern = (
        np.where((df['ema_5'] > df['ema_20']) & (df['ema_5'] > df['ema_100']), 1.2, 
                 np.where((df['ema_5'] < df['ema_20']) & (df['ema_5'] < df['ema_100']), -1.2, 0))
    )
    
    # Check for Divergence between 5-day and 20-day EMAs
    divergence = np.where((df['ema_5'] > df['ema_5'].shift(1)) & (df['ema_20'] < df['ema_20'].shift(1)), 0.6,
                          np.where((df['ema_5'] < df['ema_5'].shift(1)) & (df['ema_20'] > df['ema_20'].shift(1)), -0.6, 0))
    
    # Final Combined Score
    final_score = combined_score + historical_pattern + divergence
    
    return final_score
