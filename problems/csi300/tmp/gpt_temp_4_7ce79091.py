import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Volatility-Adjusted Momentum with Volume Divergence alpha factor
    """
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    amount = df['amount']
    
    # Volatility-Adjusted Momentum Component
    # Multi-Timeframe Raw Returns
    mom_3d = close / close.shift(3) - 1
    mom_8d = close / close.shift(8) - 1
    mom_13d = close / close.shift(13) - 1
    mom_21d = close / close.shift(21) - 1
    
    # Volatility Scaling
    daily_returns = close.pct_change()
    vol_21d = daily_returns.rolling(window=21).std()
    
    # Volatility-adjusted momentum signals
    mom_3d_vol_adj = mom_3d / (vol_21d + 1e-8)
    mom_8d_vol_adj = mom_8d / (vol_21d + 1e-8)
    mom_13d_vol_adj = mom_13d / (vol_21d + 1e-8)
    mom_21d_vol_adj = mom_21d / (vol_21d + 1e-8)
    
    # Volume Divergence Component
    # Volume Intensity Multi-Timeframe
    vol_ma_5 = volume.rolling(window=5).mean()
    vol_ma_13 = volume.rolling(window=13).mean()
    vol_ma_21 = volume.rolling(window=21).mean()
    
    vol_intensity_5 = volume / vol_ma_5
    vol_intensity_13 = volume / vol_ma_13
    vol_intensity_21 = volume / vol_ma_21
    
    # Volume Trend Persistence
    vol_dominance_count = pd.Series(
        [sum(volume.iloc[i-7:i+1] > vol_ma_21.iloc[i-7:i+1]) 
         if i >= 7 else np.nan for i in range(len(volume))], 
        index=volume.index
    )
    vol_acceleration = volume / volume.shift(1) - 1
    
    # Multiplicative Interaction Layer
    # Momentum-Timeframe Alignment
    mom_alignment_ultra_long = mom_3d_vol_adj * mom_21d_vol_adj
    mom_alignment_short_medium = mom_8d_vol_adj * mom_13d_vol_adj
    mom_direction_consistency = np.sign(mom_3d) * np.sign(mom_8d) * np.sign(mom_21d)
    
    # Volume-Momentum Confirmation
    vol_mom_confirmation_1 = vol_intensity_13 * mom_8d_vol_adj
    vol_mom_confirmation_2 = vol_intensity_21 * mom_21d_vol_adj
    vol_trend_persistence = vol_dominance_count * np.sign(mom_8d)
    
    # Liquidity Normalization Component
    # Amount-Based Liquidity Signals
    amount_ma_21 = amount.rolling(window=21).mean()
    liquidity_intensity = amount / amount_ma_21
    liquidity_momentum = amount / amount.shift(5) - 1
    
    # Price Efficiency Measures
    daily_range_efficiency = (close - low) / (high - low + 1e-8)
    opening_gap_signal = (df['open'] - close.shift(1)) / close.shift(1)
    
    # Regime Adaptation Layer
    # Volatility Regime Detection
    vol_ma_63 = vol_21d.rolling(window=63).mean()
    vol_regime_ratio = vol_21d / (vol_ma_63 + 1e-8)
    
    # Market State Indicators
    positive_returns_count = pd.Series(
        [sum(daily_returns.iloc[i-4:i+1] > 0) 
         if i >= 4 else np.nan for i in range(len(daily_returns))], 
        index=daily_returns.index
    )
    range_expansion = (high - low) / close
    
    # Final Alpha Construction
    # Core Multiplicative Combination
    primary_factor = mom_3d_vol_adj * mom_21d_vol_adj
    secondary_factor = vol_intensity_13 * vol_dominance_count
    tertiary_factor = liquidity_intensity
    
    # Combine core components
    alpha_core = primary_factor * secondary_factor * tertiary_factor
    
    # Regime Adaptive Scaling
    # Volatility regime adjustment (inverse relationship)
    vol_regime_multiplier = 1 / (vol_regime_ratio + 1e-8)
    
    # Market state conditioning (favor consistent trends)
    trend_condition = positive_returns_count / 5
    
    # Apply regime adjustments
    alpha_final = alpha_core * vol_regime_multiplier * trend_condition
    
    return alpha_final
