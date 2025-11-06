import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Dimensional Momentum Convergence Alpha Factor
    
    Combines price momentum, volume momentum, and range efficiency momentum
    across multiple timeframes to generate a robust convergence signal.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Price Momentum Dimension
    # Short-term momentum (3-day)
    mom_short = df['close'] / df['close'].shift(3) - 1
    
    # Medium-term momentum (10-day)
    mom_medium = df['close'] / df['close'].shift(10) - 1
    
    # Exponential decay weighting for momentum
    decay_weights = np.array([0.6, 0.4])  # Higher weight for recent momentum
    mom_composite = decay_weights[0] * mom_short + decay_weights[1] * mom_medium
    
    # Volume Momentum Dimension
    # Volume momentum (5-day)
    vol_mom = df['volume'] / df['volume'].shift(5) - 1
    # Volume acceleration (change in volume momentum)
    vol_accel = vol_mom - vol_mom.shift(3)
    
    # Dynamic volume adjustment
    vol_threshold = vol_accel.rolling(window=20).quantile(0.7)
    vol_multiplier = np.where(vol_accel > vol_threshold, 1.5, 
                             np.where(vol_accel < -vol_threshold, 0.7, 1.0))
    
    # Range Efficiency Momentum Dimension
    # Net price movement over 10 days
    net_movement = df['close'] / df['close'].shift(10) - 1
    
    # Total range traveled (sum of true ranges over 10 days)
    def true_range(high, low, close_prev):
        return np.maximum(high - low, 
                         np.maximum(abs(high - close_prev), 
                                   abs(low - close_prev)))
    
    tr_sum = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if i >= 10:
            tr_values = []
            for j in range(i-9, i+1):
                if j > 0:
                    tr_val = true_range(df['high'].iloc[j], df['low'].iloc[j], 
                                       df['close'].iloc[j-1])
                    tr_values.append(tr_val)
            tr_sum.iloc[i] = np.sum(tr_values) if tr_values else np.nan
    
    # Efficiency ratio
    efficiency = abs(net_movement) / (tr_sum / df['close'].shift(10))
    efficiency = efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Volume-weighted efficiency momentum
    eff_mom = net_movement * efficiency * (df['volume'] / df['volume'].rolling(20).mean())
    
    # Multi-Dimensional Convergence Scoring
    # Normalize each dimension
    def zscore_normalize(series, window=20):
        return (series - series.rolling(window).mean()) / series.rolling(window).std()
    
    mom_norm = zscore_normalize(mom_composite)
    vol_norm = zscore_normalize(vol_mom * vol_multiplier)
    eff_norm = zscore_normalize(eff_mom)
    
    # Convergence strength measurement
    convergence_strength = (mom_norm * vol_norm * eff_norm).abs()
    
    # Direction alignment score (-1 to 1)
    def sign_agreement(a, b, c):
        signs = np.sign(a) + np.sign(b) + np.sign(c)
        return signs / 3.0
    
    direction_alignment = sign_agreement(mom_norm, vol_norm, eff_norm)
    
    # Final convergence alpha
    alpha = convergence_strength * direction_alignment * mom_composite
    
    # Smooth the final signal
    alpha = alpha.rolling(window=5).mean()
    
    return alpha.fillna(0)
