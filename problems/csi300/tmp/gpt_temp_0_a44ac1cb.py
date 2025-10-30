import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Price Momentum
    data['momentum_short'] = data['close'] / data['close'].shift(2) - 1
    data['momentum_medium'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_long'] = data['close'] / data['close'].shift(10) - 1
    
    # Volume Flow Convergence
    data['money_flow'] = data['amount'] / data['volume']
    data['money_flow_avg_3d'] = data['money_flow'].rolling(window=3, min_periods=3).mean()
    data['money_flow_ratio'] = data['money_flow'] / data['money_flow_avg_3d']
    
    data['volume_avg_5d'] = data['volume'].rolling(window=5, min_periods=5).mean()
    data['volume_momentum'] = data['volume'] / data['volume_avg_5d'] - 1
    
    data['volume_strength'] = data['money_flow_ratio'] * (data['volume_momentum'] + 1)
    
    # Momentum Convergence Score
    data['momentum_positive_count'] = (
        (data['momentum_short'] > 0).astype(int) * 0.5 +
        (data['momentum_medium'] > 0).astype(int) * 0.3 +
        (data['momentum_long'] > 0).astype(int) * 0.2
    )
    
    data['weighted_momentum_score'] = (
        data['momentum_short'] * 0.5 +
        data['momentum_medium'] * 0.3 +
        data['momentum_long'] * 0.2
    )
    
    # Volume-Momentum Coherence
    convergence_factor = data['weighted_momentum_score'].copy()
    
    # Apply volume-momentum coherence adjustments
    mask_bullish_aligned = (data['volume_strength'] > 1) & (data['weighted_momentum_score'] > 0)
    mask_bearish_aligned = (data['volume_strength'] < 1) & (data['weighted_momentum_score'] < 0)
    mask_bullish_divergence = (data['volume_strength'] > 1) & (data['weighted_momentum_score'] < 0)
    mask_bearish_divergence = (data['volume_strength'] < 1) & (data['weighted_momentum_score'] > 0)
    
    convergence_factor[mask_bullish_aligned] *= 1.5
    convergence_factor[mask_bearish_aligned] *= 1.5
    convergence_factor[mask_bullish_divergence] *= 0.5
    convergence_factor[mask_bearish_divergence] *= 0.5
    
    # Price Range Confirmation
    data['true_range_pct'] = (data['high'] - data['low']) / data['close'].shift(1)
    data['true_range_avg_5d'] = data['true_range_pct'].rolling(window=5, min_periods=5).mean()
    data['range_ratio'] = data['true_range_pct'] / data['true_range_avg_5d']
    
    mask_stable_range = (data['range_ratio'] >= 0.8) & (data['range_ratio'] <= 1.2)
    mask_volatile_range = (data['range_ratio'] < 0.8) | (data['range_ratio'] > 1.2)
    
    convergence_factor[mask_stable_range] *= 1.2
    convergence_factor[mask_volatile_range] *= 0.8
    
    # Intraday Price Action
    data['opening_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_magnitude'] = abs(data['opening_gap'])
    
    # Gap filling behavior
    mask_gap_up_positive = (data['opening_gap'] > 0) & (data['close'] > data['open'])
    mask_gap_down_negative = (data['opening_gap'] < 0) & (data['close'] < data['open'])
    mask_gap_up_negative = (data['opening_gap'] > 0) & (data['close'] < data['open'])
    mask_gap_down_positive = (data['opening_gap'] < 0) & (data['close'] > data['open'])
    
    convergence_factor[mask_gap_up_positive] *= 1.3
    convergence_factor[mask_gap_down_negative] *= 1.3
    convergence_factor[mask_gap_up_negative] *= 0.7
    convergence_factor[mask_gap_down_positive] *= 0.7
    
    # Dynamic Smoothing with 3-day exponential weighting
    weights = [0.6, 0.3, 0.1]
    smoothed_factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if i >= 2:
            window_values = convergence_factor.iloc[i-2:i+1]
            smoothed_factor.iloc[i] = np.average(window_values, weights=weights)
        else:
            smoothed_factor.iloc[i] = convergence_factor.iloc[i]
    
    return smoothed_factor
