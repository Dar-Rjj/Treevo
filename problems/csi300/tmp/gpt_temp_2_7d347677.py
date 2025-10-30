import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Multi-Timeframe Momentum Alignment
    # Short-term Momentum (2-day)
    mom_short = df['close'] / df['close'].shift(2) - 1
    
    # Medium-term Momentum (5-day)
    mom_medium = df['close'] / df['close'].shift(5) - 1
    
    # Long-term Momentum (10-day)
    mom_long = df['close'] / df['close'].shift(10) - 1
    
    # Momentum Convergence Score
    sign_alignment = ((mom_short > 0) & (mom_medium > 0) & (mom_long > 0)) | ((mom_short < 0) & (mom_medium < 0) & (mom_long < 0))
    strength = np.abs(mom_short) * np.abs(mom_medium) * np.abs(mom_long)
    direction = np.where((mom_short > 0) & (mom_medium > 0) & (mom_long > 0), 1, 
                        np.where((mom_short < 0) & (mom_medium < 0) & (mom_long < 0), -1, 0))
    
    momentum_convergence = sign_alignment.astype(float) * strength * direction
    
    # Volume Confirmation Signals
    # Short-term Volume Change
    vol_short = df['volume'] / df['volume'].shift(2)
    
    # Medium-term Volume Change
    vol_medium = df['volume'] / df['volume'].shift(5)
    
    # Volume Confirmation Score
    direction_match = ((momentum_convergence > 0) & (vol_short > 1) & (vol_medium > 1)) | ((momentum_convergence < 0) & (vol_short < 1) & (vol_medium < 1))
    vol_strength = (vol_short + vol_medium) / 2
    confirmation_flag = direction_match
    
    # Volatility Context Assessment
    # Short-term Volatility
    short_vol = (df['high'] - df['low']) / df['close']
    
    # Medium-term Volatility
    medium_vol = pd.Series([(df['high'].iloc[i-9:i+1] - df['low'].iloc[i-9:i+1]).div(df['close'].iloc[i-9:i+1]).mean() 
                           for i in range(9, len(df))], index=df.index[9:])
    
    # Volatility Regime
    volatility_ratio = short_vol / medium_vol
    
    # Composite Alpha Factor
    # Base Momentum Component
    raw_momentum = momentum_convergence
    
    # Volume Enhancement
    volume_multiplier = np.where(confirmation_flag, 1.5, 0.7)
    
    # Volatility Filter
    low_vol_regime = volatility_ratio < 0.8
    high_vol_regime = volatility_ratio > 1.2
    
    # Base Value
    base_factor = raw_momentum * volume_multiplier
    
    # Final Alpha Factor with Volatility Adjustment
    final_alpha = pd.Series(index=df.index, dtype=float)
    
    # Apply volatility scaling only in low volatility regime
    final_alpha[low_vol_regime] = base_factor[low_vol_regime] * (1 / volatility_ratio[low_vol_regime])
    
    # Zero output in high volatility regimes
    final_alpha[high_vol_regime] = 0
    
    # For normal volatility regimes, use base factor
    normal_vol_regime = ~low_vol_regime & ~high_vol_regime
    final_alpha[normal_vol_regime] = base_factor[normal_vol_regime]
    
    return final_alpha
