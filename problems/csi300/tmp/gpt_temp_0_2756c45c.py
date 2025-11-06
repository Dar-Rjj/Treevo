import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Asymmetric Return Distribution Analysis
    upside_momentum = returns.rolling(window=5, min_periods=1).apply(
        lambda x: x[x > 0].sum(), raw=False
    )
    downside_momentum = returns.rolling(window=5, min_periods=1).apply(
        lambda x: abs(x[x < 0].sum()), raw=False
    )
    momentum_asymmetry = upside_momentum / (downside_momentum + 1e-8)
    
    # Volume Participation Dynamics
    buy_volume = df['volume'].where(returns > 0, 0)
    sell_volume = df['volume'].where(returns < 0, 0)
    
    buy_volume_5d = buy_volume.rolling(window=5, min_periods=1).sum()
    sell_volume_5d = sell_volume.rolling(window=5, min_periods=1).sum()
    volume_participation = buy_volume_5d / (sell_volume_5d + 1e-8)
    
    # Price-Volume Divergence Detection
    price_low_5d = df['low'].rolling(window=5, min_periods=1).min()
    sell_volume_5d_min = sell_volume.rolling(window=5, min_periods=1).min()
    
    price_high_5d = df['high'].rolling(window=5, min_periods=1).max()
    buy_volume_5d_min = buy_volume.rolling(window=5, min_periods=1).min()
    
    # Bullish divergence: lower price lows with decreasing sell volume
    bullish_divergence = ((df['low'] == price_low_5d) & 
                         (sell_volume == sell_volume_5d_min)).astype(int)
    
    # Bearish divergence: higher price highs with decreasing buy volume
    bearish_divergence = ((df['high'] == price_high_5d) & 
                         (buy_volume == buy_volume_5d_min)).astype(int)
    
    divergence_strength = bullish_divergence - bearish_divergence
    
    # Multi-timeframe Pattern Confirmation
    def calculate_divergence_pattern(window):
        price_low_window = df['low'].rolling(window=window, min_periods=1).min()
        sell_volume_window_min = sell_volume.rolling(window=window, min_periods=1).min()
        price_high_window = df['high'].rolling(window=window, min_periods=1).max()
        buy_volume_window_min = buy_volume.rolling(window=window, min_periods=1).min()
        
        bullish_pattern = ((df['low'] == price_low_window) & 
                          (sell_volume == sell_volume_window_min)).astype(int)
        bearish_pattern = ((df['high'] == price_high_window) & 
                          (buy_volume == buy_volume_window_min)).astype(int)
        
        return bullish_pattern - bearish_pattern
    
    # Calculate divergence patterns for different timeframes
    short_term_divergence = calculate_divergence_pattern(3)
    medium_term_divergence = calculate_divergence_pattern(8)
    long_term_divergence = calculate_divergence_pattern(15)
    
    # Combine all components into final factor
    factor = (momentum_asymmetry * 0.3 + 
              volume_participation * 0.2 + 
              divergence_strength * 0.2 +
              short_term_divergence * 0.1 +
              medium_term_divergence * 0.1 +
              long_term_divergence * 0.1)
    
    return factor
