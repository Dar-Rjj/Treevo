import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Momentum Velocity with Volatility-Weighted Efficiency and Squeeze Detection
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Velocity Analysis
    # Calculate 3-day, 8-day, and 21-day price momentum
    momentum_3d = data['close'] / data['close'].shift(3) - 1
    momentum_8d = data['close'] / data['close'].shift(8) - 1
    momentum_21d = data['close'] / data['close'].shift(21) - 1
    
    # Compute momentum velocity (acceleration) between timeframes
    velocity_3_8 = momentum_3d - momentum_8d
    velocity_8_21 = momentum_8d - momentum_21d
    velocity_3_21 = momentum_3d - momentum_21d
    
    # Apply exponential decay weighting (λ = 0.94)
    lambda_val = 0.94
    weights = np.array([lambda_val**i for i in range(5)])[::-1]
    weights = weights / weights.sum()
    
    def exp_weighted_mean(series):
        if len(series) >= len(weights):
            return np.convolve(series.dropna(), weights, mode='valid')[-1]
        else:
            return series.mean()
    
    momentum_velocity = (
        exp_weighted_mean(pd.Series([velocity_3_8.iloc[-1], velocity_8_21.iloc[-1], velocity_3_21.iloc[-1]])) 
        if len(data) >= 5 else 0
    )
    
    # Detect momentum divergence patterns
    momentum_divergence = (
        np.sign(momentum_3d) * np.sign(momentum_8d) * np.sign(momentum_21d) * 
        (abs(momentum_3d) + abs(momentum_8d) + abs(momentum_21d)) / 3
    )
    
    # Volume-Amount Efficiency with Volatility Scaling
    # Calculate 5-day and 10-day volume momentum
    volume_5d = data['volume'].rolling(window=5).mean()
    volume_10d = data['volume'].rolling(window=10).mean()
    volume_momentum = volume_5d / volume_10d - 1
    
    # Compute price change per unit volume (efficiency metric)
    price_change = data['close'].pct_change()
    volume_efficiency = price_change / (data['volume'].replace(0, np.nan) / data['volume'].rolling(window=20).mean())
    
    # Assess efficiency trend
    efficiency_short = volume_efficiency.rolling(window=5).mean()
    efficiency_medium = volume_efficiency.rolling(window=10).mean()
    efficiency_trend = efficiency_short / efficiency_medium - 1
    
    # Incorporate amount data for order flow context
    amount_efficiency = price_change / (data['amount'].replace(0, np.nan) / data['amount'].rolling(window=20).mean())
    
    # Apply volatility adjustment using True Range (ATR)
    def calculate_atr(data, window):
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift(1))
        low_close = abs(data['low'] - data['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    atr_14 = calculate_atr(data, 14)
    volatility_scaled_efficiency = volume_efficiency / atr_14.replace(0, np.nan)
    
    # Volatility Compression and Regime Classification
    # Bollinger Band width analysis
    bb_ma_20 = data['close'].rolling(window=20).mean()
    bb_std_20 = data['close'].rolling(window=20).std()
    bb_width = (bb_std_20 / bb_ma_20).replace(0, np.nan)
    
    # Volatility compression detection
    vol_short = data['close'].pct_change().rolling(window=5).std()
    vol_medium = data['close'].pct_change().rolling(window=20).std()
    vol_compression = vol_short / vol_medium
    
    # Range constriction analysis
    daily_range = data['high'] - data['low']
    range_5d = daily_range.rolling(window=5).mean()
    range_20d = daily_range.rolling(window=20).mean()
    range_constriction = range_5d / range_20d
    
    # Calculate 20-day ATR vs 60-day ATR ratio
    atr_20 = calculate_atr(data, 20)
    atr_60 = calculate_atr(data, 60)
    atr_ratio = atr_20 / atr_60.replace(0, np.nan)
    
    # Classify volatility regimes
    def classify_regime(atr_ratio_val):
        if atr_ratio_val > 1.2:
            return 'high'
        elif atr_ratio_val < 0.8:
            return 'low'
        else:
            return 'normal'
    
    volatility_regime = atr_ratio.apply(classify_regime)
    
    # Regime-Adaptive Signal Integration
    def regime_adaptive_signal(row):
        regime = volatility_regime.loc[row.name]
        
        if regime == 'high':
            # High Volatility Regime weights
            vol_scaled_eff = volatility_scaled_efficiency.loc[row.name] if not pd.isna(volatility_scaled_efficiency.loc[row.name]) else 0
            mom_vel = momentum_velocity if not pd.isna(momentum_velocity) else 0
            vol_amount_align = (volume_efficiency.loc[row.name] + amount_efficiency.loc[row.name]) / 2 if not pd.isna(volume_efficiency.loc[row.name]) else 0
            
            return (0.5 * vol_scaled_eff + 
                   0.3 * mom_vel + 
                   0.2 * vol_amount_align)
        
        elif regime == 'low':
            # Low Volatility Regime weights
            mom_vel = momentum_velocity if not pd.isna(momentum_velocity) else 0
            eff_trend = efficiency_trend.loc[row.name] if not pd.isna(efficiency_trend.loc[row.name]) else 0
            vol_comp = vol_compression.loc[row.name] if not pd.isna(vol_compression.loc[row.name]) else 0
            
            return (0.5 * mom_vel + 
                   0.3 * eff_trend + 
                   0.2 * (1 - vol_comp))  # Inverse of compression for squeeze potential
        
        else:
            # Normal Regime weights
            mom_vel = momentum_velocity if not pd.isna(momentum_velocity) else 0
            eff_trend = efficiency_trend.loc[row.name] if not pd.isna(efficiency_trend.loc[row.name]) else 0
            
            # Volume distribution and amount concentration
            volume_dist = (data['volume'].loc[row.name] / data['volume'].rolling(window=20).mean().loc[row.name] 
                          if not pd.isna(data['volume'].rolling(window=20).mean().loc[row.name]) else 1)
            amount_conc = (data['amount'].loc[row.name] / data['amount'].rolling(window=20).mean().loc[row.name] 
                          if not pd.isna(data['amount'].rolling(window=20).mean().loc[row.name]) else 1)
            
            return (0.4 * mom_vel + 
                   0.4 * eff_trend + 
                   0.1 * volume_dist + 
                   0.1 * amount_conc)
    
    regime_signal = pd.Series(index=data.index, dtype=float)
    for idx in data.index:
        regime_signal.loc[idx] = regime_adaptive_signal(data.loc[idx:idx])
    
    # Squeeze and Breakout Pattern Detection
    # Early expansion signal identification
    range_expansion = range_5d / range_20d
    volume_expansion = volume_5d / volume_10d
    
    # Squeeze intensity multiplier
    squeeze_intensity = (1 / bb_width.replace(0, np.nan)) * (1 - vol_compression)
    
    # Detect alignment between momentum velocity and efficiency
    momentum_efficiency_alignment = (
        np.sign(momentum_velocity) * np.sign(efficiency_trend) * 
        (abs(momentum_velocity) + abs(efficiency_trend)) / 2
    )
    
    # Volatility Context Enhancement
    # Scale signals by 14-day ATR for volatility adjustment
    atr_scaling = atr_14 / atr_14.rolling(window=20).mean()
    
    # Apply Bollinger Band squeeze amplification
    bb_squeeze_amp = (1 / bb_width.replace(0, np.nan)) * squeeze_intensity
    
    # Composite Alpha Factor Generation
    # Combine regime-weighted components with volatility context
    base_signal = regime_signal
    
    # Apply volatility context enhancement
    volatility_adjusted_signal = base_signal * atr_scaling.replace(0, np.nan)
    
    # Incorporate squeeze and breakout patterns
    squeeze_boost = squeeze_intensity * momentum_efficiency_alignment
    breakout_signal = range_expansion * volume_expansion * momentum_efficiency_alignment
    
    # Final composite alpha factor
    alpha_factor = (
        volatility_adjusted_signal * 0.6 +
        squeeze_boost * 0.25 +
        breakout_signal * 0.15
    )
    
    # Clean and return the factor
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha_factor
