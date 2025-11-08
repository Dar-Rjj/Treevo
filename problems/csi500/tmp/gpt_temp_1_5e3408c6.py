import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns for volatility
    returns = df['close'].pct_change()
    
    # 5-Day Analysis
    price_momentum_5 = df['close'] / df['close'].shift(5)
    volume_momentum_5 = df['volume'] / df['volume'].shift(5)
    divergence_5 = price_momentum_5 - volume_momentum_5
    
    # 20-Day Analysis
    price_momentum_20 = df['close'] / df['close'].shift(20)
    volume_momentum_20 = df['volume'] / df['volume'].shift(20)
    divergence_20 = price_momentum_20 - volume_momentum_20
    
    # Dynamic Volatility Regime Weighting
    volatility_20 = returns.rolling(window=20, min_periods=1).std()
    volatility_percentile = volatility_20.rolling(window=50, min_periods=1).apply(
        lambda x: (x.iloc[-1] > x).mean(), raw=False
    )
    
    # Initialize weights
    weight_5 = pd.Series(0.5, index=df.index)
    weight_20 = pd.Series(0.5, index=df.index)
    
    # Assign weights based on volatility regime
    high_vol_mask = volatility_percentile > 0.7
    low_vol_mask = volatility_percentile < 0.3
    
    weight_5[high_vol_mask] = 0.2
    weight_20[high_vol_mask] = 0.8
    
    weight_5[low_vol_mask] = 0.8
    weight_20[low_vol_mask] = 0.2
    
    # Adaptive Volume Spike Detection
    volume_median_20 = df['volume'].rolling(window=20, min_periods=1).median()
    volume_threshold = 2 * volume_median_20
    volume_spike_multiplier = np.where(df['volume'] > volume_threshold, 1.5, 1.0)
    
    # Price Level Context Integration
    high_20 = df['high'].rolling(window=20, min_periods=1).max()
    low_20 = df['low'].rolling(window=20, min_periods=1).min()
    price_position = (df['close'] - low_20) / (high_20 - low_20)
    
    # Initialize position multiplier
    position_multiplier = pd.Series(1.0, index=df.index)
    
    # Assign position multipliers
    resistance_mask = price_position > 0.8
    support_mask = price_position < 0.2
    
    position_multiplier[resistance_mask] = 1.3
    position_multiplier[support_mask] = 0.7
    
    # Final Alpha Factor Construction
    weighted_divergence = (divergence_5 * weight_5) + (divergence_20 * weight_20)
    final_factor = weighted_divergence * volume_spike_multiplier * position_multiplier
    
    return final_factor
