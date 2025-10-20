import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Period Momentum Calculation
    # Compute short-term momentum (5-day close return)
    short_momentum = data['close'].pct_change(periods=5)
    
    # Compute medium-term momentum (20-day close return)
    medium_momentum = data['close'].pct_change(periods=20)
    
    # Apply exponential decay weighting to both momentum series
    def apply_exponential_decay(series, window=20):
        weights = np.exp(-np.arange(window) / 5)  # Decay factor of 5
        weights = weights / weights.sum()
        
        decayed = pd.Series(index=series.index, dtype=float)
        for i in range(len(series)):
            if i >= window - 1:
                window_data = series.iloc[i-window+1:i+1]
                decayed.iloc[i] = (window_data * weights).sum()
            else:
                decayed.iloc[i] = np.nan
        return decayed
    
    short_momentum_decayed = apply_exponential_decay(short_momentum, window=5)
    medium_momentum_decayed = apply_exponential_decay(medium_momentum, window=20)
    
    # Volatility-Based Adjustment
    # Calculate Average True Range (14-day)
    high_low = data['high'] - data['low']
    high_close_prev = abs(data['high'] - data['close'].shift(1))
    low_close_prev = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    atr_14 = true_range.rolling(window=14).mean()
    
    # Calculate daily range volatility (High - Low) 14-day average
    range_volatility = (data['high'] - data['low']).rolling(window=14).mean()
    
    # Compute combined volatility measure
    combined_volatility = (atr_14 + range_volatility) / 2
    
    # Divide both momentum signals by combined volatility
    short_momentum_vol_adj = short_momentum_decayed / combined_volatility
    medium_momentum_vol_adj = medium_momentum_decayed / combined_volatility
    
    # Bollinger Squeeze Detection
    # Calculate Bollinger Band width (Upper Band - Lower Band) using 20-day MA and 2σ
    bb_ma = data['close'].rolling(window=20).mean()
    bb_std = data['close'].rolling(window=20).std()
    bb_upper = bb_ma + 2 * bb_std
    bb_lower = bb_ma - 2 * bb_std
    bb_width = bb_upper - bb_lower
    
    # Compute squeeze intensity as inverse of band width to MA ratio
    squeeze_intensity = 1 / (bb_width / bb_ma)
    
    # Momentum Divergence Analysis
    # Detect momentum divergence conditions
    divergence_condition = (short_momentum_vol_adj > 0) & (medium_momentum_vol_adj < 0)
    
    # Calculate divergence strength
    divergence_strength = abs(short_momentum_vol_adj - medium_momentum_vol_adj)
    divergence_strength = divergence_strength / data['close']  # Scale by recent price level
    
    # Volume Confirmation Analysis
    # Calculate volume trend ratio (5-day average volume / 20-day average volume)
    volume_trend_ratio = (data['volume'].rolling(window=5).mean() / 
                         data['volume'].rolling(window=20).mean())
    
    # Compute volume acceleration
    volume_first_diff = data['volume'].diff()
    volume_second_diff = volume_first_diff.diff()
    
    # Integrated Alpha Signal Generation
    # Combine volatility-adjusted momentum divergence with squeeze intensity
    divergence_component = divergence_strength * squeeze_intensity
    
    # Apply exponential weighting to divergence component
    divergence_weighted = apply_exponential_decay(divergence_component, window=10)
    
    # Integrate volume confirmation signals
    volume_acceleration_factor = 1 + (volume_second_diff / data['volume'].rolling(window=10).mean())
    volume_confirmation = volume_trend_ratio * volume_acceleration_factor
    
    # Generate final alpha factor
    alpha_factor = divergence_weighted * volume_confirmation
    
    # Apply divergence condition filter
    alpha_factor = alpha_factor.where(divergence_condition, 0)
    
    return alpha_factor
