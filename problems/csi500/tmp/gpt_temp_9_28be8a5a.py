import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Multi-Timeframe Momentum-Range Efficiency factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Convergence
    # Short-term Momentum (3-day)
    mom_short = data['close'] / data['close'].shift(3) - 1
    
    # Medium-term Momentum (5-day)
    mom_medium = data['close'] / data['close'].shift(5) - 1
    
    # Long-term Momentum (10-day)
    mom_long = data['close'] / data['close'].shift(10) - 1
    
    # Convergence Strength Analysis
    # Directional alignment across all three timeframes
    directional_alignment = ((mom_short > 0) & (mom_medium > 0) & (mom_long > 0)).astype(int) - \
                           ((mom_short < 0) & (mom_medium < 0) & (mom_long < 0)).astype(int)
    
    # Momentum gradient: (Short-Medium) vs (Medium-Long) differences
    mom_gradient = (mom_short - mom_medium) - (mom_medium - mom_long)
    
    # Convergence persistence: 3-day consistency count
    mom_consistency = pd.Series(0, index=data.index)
    for i in range(3, len(data)):
        window = data.iloc[i-2:i+1]
        if len(window) == 3:
            short_vals = [window['close'].iloc[j] / window['close'].iloc[max(0, j-3)] - 1 for j in range(1, 3)]
            med_vals = [window['close'].iloc[j] / window['close'].iloc[max(0, j-5)] - 1 for j in range(1, 3)]
            long_vals = [window['close'].iloc[j] / window['close'].iloc[max(0, j-10)] - 1 for j in range(1, 3)]
            
            consistent = sum(1 for s, m, l in zip(short_vals, med_vals, long_vals) 
                           if (s > 0 and m > 0 and l > 0) or (s < 0 and m < 0 and l < 0))
            mom_consistency.iloc[i] = consistent
    
    convergence_strength = directional_alignment * (1 + abs(mom_gradient)) * (1 + mom_consistency / 3)
    
    # Range Efficiency Analysis
    # True Range Calculation
    tr1 = data['high'] - data['low']
    tr2 = abs(data['high'] - data['close'].shift(1))
    tr3 = abs(data['low'] - data['close'].shift(1))
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Price Movement Efficiency
    price_efficiency = (data['close'] - data['open']) / true_range.replace(0, np.nan)
    price_efficiency = price_efficiency.fillna(0)
    
    # Range Deviation: Current range vs 5-day range average
    range_5d_avg = true_range.rolling(window=5, min_periods=3).mean()
    range_deviation = true_range / range_5d_avg.replace(0, np.nan) - 1
    range_deviation = range_deviation.fillna(0)
    
    # Volume-Weighted Divergence Detection
    # Volume Momentum Analysis
    vol_mom_short = data['volume'] / data['volume'].shift(3).replace(0, np.nan) - 1
    vol_mom_medium = data['volume'] / data['volume'].shift(5).replace(0, np.nan) - 1
    vol_mom_short = vol_mom_short.fillna(0)
    vol_mom_medium = vol_mom_medium.fillna(0)
    
    # Price-Volume Alignment
    price_vol_alignment = ((mom_short > 0) & (vol_mom_short > 0)).astype(int) - \
                         ((mom_short < 0) & (vol_mom_short < 0)).astype(int)
    
    # Volume confirmation strength: absolute volume momentum
    vol_confirmation = (abs(vol_mom_short) + abs(vol_mom_medium)) / 2
    
    # Efficiency-Weighted Divergence
    volume_divergence = price_vol_alignment * vol_confirmation * price_efficiency
    
    # Volume persistence (3-day volume trend)
    vol_persistence = data['volume'].rolling(window=3, min_periods=2).apply(
        lambda x: 1 if len(x) >= 2 and x.iloc[-1] > x.iloc[0] else -1 if len(x) >= 2 and x.iloc[-1] < x.iloc[0] else 0, 
        raw=False
    )
    
    volume_weighted_divergence = volume_divergence * (1 + abs(vol_persistence))
    
    # Regime-Adaptive Signal Construction
    # Volatility Regime Detection
    range_20d_median = true_range.rolling(window=20, min_periods=10).median()
    atr_10d = true_range.rolling(window=10, min_periods=5).mean()
    
    # Regime classification
    high_vol_regime = (true_range > range_20d_median * 1.2) & (atr_10d > atr_10d.rolling(window=20, min_periods=10).median())
    low_vol_regime = (true_range < range_20d_median * 0.8) & (atr_10d < atr_10d.rolling(window=20, min_periods=10).median())
    transition_regime = ~high_vol_regime & ~low_vol_regime
    
    # Regime persistence: 8-day volatility consistency
    vol_consistency = true_range.rolling(window=8, min_periods=5).std() / true_range.rolling(window=8, min_periods=5).mean()
    
    # Signal Adjustment by Regime
    raw_signal = convergence_strength * range_deviation * (1 + volume_weighted_divergence)
    
    regime_multiplier = pd.Series(1.0, index=data.index)
    regime_multiplier[high_vol_regime] = 1.2
    regime_multiplier[low_vol_regime] = -0.8
    regime_multiplier[transition_regime] = 0.6 + (0.4 * vol_consistency[transition_regime])
    
    # Final factor calculation
    factor = raw_signal * regime_multiplier
    
    # Mean-Relative Final Output
    factor_15d_mean = factor.rolling(window=15, min_periods=10).mean()
    final_factor = factor - factor_15d_mean
    
    return final_factor
