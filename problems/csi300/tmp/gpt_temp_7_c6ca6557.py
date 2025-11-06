import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate returns
    returns = df['close'].pct_change()
    
    # Volatility Regime Classification
    short_vol = returns.rolling(window=5).std()
    medium_vol = returns.rolling(window=20).std()
    vol_regime_ratio = short_vol / medium_vol
    
    # Regime Persistence Analysis
    vol_trend = (vol_regime_ratio > vol_regime_ratio.shift(1)).rolling(window=5).sum()
    regime_stability = vol_regime_ratio.rolling(window=10).std()
    
    # Microstructure Efficiency Components
    # Range-Based Efficiency Metrics
    intraday_range = df['high'] - df['low']
    intraday_range = intraday_range.replace(0, np.nan)  # Avoid division by zero
    
    range_efficiency = (df['close'] - df['open']) / intraday_range
    price_positioning = (df['close'] - df['low']) / intraday_range
    opening_efficiency = abs(df['open'] - (df['high'] + df['low']) / 2) / intraday_range
    
    # Volume-Efficiency Alignment
    volume_acceleration = df['volume'] / df['volume'].shift(1)
    vol_change = short_vol / short_vol.shift(1)
    volume_volatility_elasticity = volume_acceleration / vol_change.replace(0, np.nan)
    
    # Calculate rolling correlation between range efficiency and volume
    efficiency_volume_corr = pd.Series(index=df.index, dtype=float)
    for i in range(5, len(df)):
        window_data = df.iloc[i-5:i]
        if len(window_data) >= 5:
            corr_val = window_data['volume'].corr(range_efficiency.iloc[i-5:i])
            efficiency_volume_corr.iloc[i] = corr_val if not np.isnan(corr_val) else 0
    
    # Microstructure Noise Assessment
    vwap_proxy = df['amount'] / df['volume'].replace(0, np.nan)
    price_impact = abs(df['close'] - df['open']) / vwap_proxy
    price_discreteness = (df['close'] * 100 % 1) / intraday_range
    composite_noise = (price_impact + price_discreteness) / 2
    
    # Momentum Construction with Regime Adaptation
    # Multi-timeframe Momentum Signals
    short_momentum = df['close'] / df['close'].shift(3) - 1
    medium_momentum = df['close'] / df['close'].shift(6) - 1
    momentum_alignment = np.sign(short_momentum) * np.sign(medium_momentum)
    
    # Efficiency-Enhanced Momentum Filtering
    range_efficiency_momentum = short_momentum * range_efficiency
    volume_confirmed_momentum = short_momentum * volume_acceleration
    noise_adjusted_momentum = short_momentum * (1 - composite_noise)
    
    # Regime-Adaptive Momentum Scaling
    high_vol_regime = vol_regime_ratio > 1.2
    low_vol_regime = vol_regime_ratio < 0.8
    
    regime_adaptive_momentum = pd.Series(index=df.index, dtype=float)
    for i in range(len(df)):
        if high_vol_regime.iloc[i] if not pd.isna(high_vol_regime.iloc[i]) else False:
            regime_adaptive_momentum.iloc[i] = noise_adjusted_momentum.iloc[i]
        elif low_vol_regime.iloc[i] if not pd.isna(low_vol_regime.iloc[i]) else False:
            regime_adaptive_momentum.iloc[i] = range_efficiency_momentum.iloc[i]
        else:
            # Blend for transition regimes
            regime_adaptive_momentum.iloc[i] = (
                0.4 * range_efficiency_momentum.iloc[i] + 
                0.3 * volume_confirmed_momentum.iloc[i] + 
                0.3 * noise_adjusted_momentum.iloc[i]
            )
    
    # Price-Volume Divergence Detection
    price_sign = np.sign(short_momentum)
    volume_sign = np.sign(volume_acceleration - 1)
    directional_divergence = (price_sign != volume_sign).astype(int)
    
    price_magnitude = abs(short_momentum)
    volume_magnitude = abs(volume_acceleration - 1)
    speed_divergence = price_magnitude / volume_magnitude.replace(0, np.nan)
    
    # Adaptive Alpha Synthesis
    # Core Factor Construction
    efficiency_weighted_momentum = regime_adaptive_momentum * range_efficiency
    volume_confirmation_multiplier = efficiency_weighted_momentum * volume_acceleration
    noise_reduced_factor = volume_confirmation_multiplier * (1 - composite_noise)
    
    # Divergence-Based Adjustments
    divergence_adjusted_factor = noise_reduced_factor.copy()
    
    # Apply divergence adjustments based on regime
    for i in range(len(df)):
        if directional_divergence.iloc[i]:
            if high_vol_regime.iloc[i] if not pd.isna(high_vol_regime.iloc[i]) else False:
                # In high vol, speed divergence is more important
                if speed_divergence.iloc[i] > 2:
                    divergence_adjusted_factor.iloc[i] *= 0.7
                elif speed_divergence.iloc[i] < 0.5:
                    divergence_adjusted_factor.iloc[i] *= 1.2
            else:
                # In low vol, directional divergence is more important
                divergence_adjusted_factor.iloc[i] *= 0.8
    
    # Regime-Confidence Integration
    vol_trend_strength = vol_trend / 5  # Normalize to 0-1
    efficiency_persistence = (1 - opening_efficiency.rolling(window=5).std())
    
    final_alpha = (
        divergence_adjusted_factor * 
        vol_trend_strength * 
        efficiency_persistence
    )
    
    return final_alpha
