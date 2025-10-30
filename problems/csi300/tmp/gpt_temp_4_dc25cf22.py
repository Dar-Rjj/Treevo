import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Reversal Detection
    # Recent price extremes
    data['high_10'] = data['high'].rolling(window=10, min_periods=5).max()
    data['low_10'] = data['low'].rolling(window=10, min_periods=5).min()
    
    # Momentum divergence (5-day vs 20-day)
    data['momentum_5'] = data['close'].pct_change(5)
    data['momentum_20'] = data['close'].pct_change(20)
    data['momentum_divergence'] = np.where(
        (data['momentum_5'] > 0) & (data['momentum_20'] < 0), 1,
        np.where((data['momentum_5'] < 0) & (data['momentum_20'] > 0), -1, 0)
    )
    
    # Breakout reversal patterns
    data['breakout_up'] = np.where(
        (data['close'] > data['high_10'].shift(1)) & 
        (data['momentum_divergence'] == -1), -1, 0
    )
    data['breakout_down'] = np.where(
        (data['close'] < data['low_10'].shift(1)) & 
        (data['momentum_divergence'] == 1), 1, 0
    )
    data['reversal_signal'] = data['breakout_up'] + data['breakout_down']
    
    # Volatility Analysis
    # True Range calculation
    data['tr'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['atr_10'] = data['tr'].rolling(window=10, min_periods=5).mean()
    data['tr_expansion'] = (data['tr'] - data['atr_10']) / data['atr_10']
    
    # Volatility-adjusted momentum
    data['volatility_20'] = data['close'].pct_change().rolling(window=20, min_periods=10).std()
    data['momentum_5_vol_adj'] = data['momentum_5'] / (data['volatility_20'] + 1e-8)
    data['momentum_20_vol_adj'] = data['momentum_20'] / (data['volatility_20'] + 1e-8)
    
    # Volume Confirmation
    # Volume trend analysis
    data['volume_slope_5'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    data['volume_slope_10'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    
    # Volume-price divergence
    data['price_slope_5'] = data['close'].rolling(window=5, min_periods=3).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
    )
    data['volume_divergence'] = np.where(
        (data['price_slope_5'] > 0) & (data['volume_slope_5'] < 0), -1,
        np.where((data['price_slope_5'] < 0) & (data['volume_slope_5'] > 0), 1, 0)
    )
    
    # Factor Integration
    # Combine reversal and volatility signals
    data['reversal_vol_combo'] = data['reversal_signal'] * (1 + data['tr_expansion'])
    
    # Apply volume divergence multiplier
    volume_multiplier = 1 + (0.5 * data['volume_divergence'])
    
    # Generate final alpha factor
    alpha_factor = (
        data['reversal_vol_combo'] * 
        (data['momentum_5_vol_adj'] - data['momentum_20_vol_adj']) * 
        volume_multiplier
    )
    
    return alpha_factor
