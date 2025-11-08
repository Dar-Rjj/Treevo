import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    """
    Regime-Adaptive Momentum-Volume Congruence Factor
    """
    df = data.copy()
    
    # Multi-Timeframe Momentum Analysis
    # Short-term (5-day)
    df['price_mom_5'] = df['close'] / df['close'].shift(5) - 1
    df['volume_mom_5'] = df['volume'] / df['volume'].shift(5) - 1
    
    # Medium-term (10-day)
    df['price_mom_10'] = df['close'] / df['close'].shift(10) - 1
    df['volume_mom_10'] = df['volume'] / df['volume'].shift(10) - 1
    
    # Long-term (20-day)
    df['price_mom_20'] = df['close'] / df['close'].shift(20) - 1
    df['volume_mom_20'] = df['volume'] / df['volume'].shift(20) - 1
    
    # Momentum Decay Weighting
    decay_weights = {
        '5': np.exp(-0.2),
        '10': np.exp(-0.1),
        '20': np.exp(-0.05)
    }
    
    # Weighted momentum calculations
    df['weighted_price_mom'] = (
        decay_weights['5'] * df['price_mom_5'] +
        decay_weights['10'] * df['price_mom_10'] +
        decay_weights['20'] * df['price_mom_20']
    )
    
    df['weighted_volume_mom'] = (
        decay_weights['5'] * df['volume_mom_5'] +
        decay_weights['10'] * df['volume_mom_10'] +
        decay_weights['20'] * df['volume_mom_20']
    )
    
    # Volume Acceleration Analysis
    # Calculate volume slopes using linear regression
    def calc_slope(series, window):
        x = np.arange(window)
        slopes = []
        for i in range(len(series) - window + 1):
            y = series.iloc[i:i+window].values
            if len(y) == window:
                slope = np.polyfit(x, y, 1)[0]
                slopes.append(slope)
            else:
                slopes.append(np.nan)
        return pd.Series(slopes, index=series.index[window-1:])
    
    df['volume_slope_5'] = calc_slope(df['volume'], 5)
    df['volume_slope_10'] = calc_slope(df['volume'], 10)
    
    # Volume acceleration (change in slope)
    df['volume_accel'] = df['volume_slope_5'] - df['volume_slope_10'].shift(5)
    
    # Congruence-Divergence Scoring
    # Sign congruence
    df['sign_congruence_5'] = (np.sign(df['price_mom_5']) == np.sign(df['volume_mom_5'])).astype(int)
    df['sign_congruence_10'] = (np.sign(df['price_mom_10']) == np.sign(df['volume_mom_10'])).astype(int)
    df['sign_congruence_20'] = (np.sign(df['price_mom_20']) == np.sign(df['volume_mom_20'])).astype(int)
    
    # Magnitude congruence (normalized ratio)
    def magnitude_congruence(price_mom, volume_mom):
        price_abs = np.abs(price_mom)
        volume_abs = np.abs(volume_mom)
        max_val = np.maximum(price_abs, volume_abs)
        return np.where(max_val > 0, np.minimum(price_abs, volume_abs) / max_val, 0)
    
    df['mag_congruence_5'] = magnitude_congruence(df['price_mom_5'], df['volume_mom_5'])
    df['mag_congruence_10'] = magnitude_congruence(df['price_mom_10'], df['volume_mom_10'])
    df['mag_congruence_20'] = magnitude_congruence(df['price_mom_20'], df['volume_mom_20'])
    
    # Timeframe consistency
    df['timeframe_consistency'] = (
        df['sign_congruence_5'] + df['sign_congruence_10'] + df['sign_congruence_20']
    ) / 3
    
    # Regime-Aware Weighting
    # Volatility regime detection
    df['price_volatility'] = df['close'].pct_change().rolling(20).std()
    df['volume_volatility'] = df['volume'].pct_change().rolling(20).std()
    
    # Normalize volatilities
    df['norm_price_vol'] = df['price_volatility'] / df['price_volatility'].rolling(60).mean()
    df['norm_volume_vol'] = df['volume_volatility'] / df['volume_volatility'].rolling(60).mean()
    
    # Market regime score (higher = more volatile)
    df['regime_score'] = (df['norm_price_vol'] + df['norm_volume_vol']) / 2
    
    # Final Factor Construction
    # Base congruence score
    base_congruence = (
        decay_weights['5'] * (df['sign_congruence_5'] * df['mag_congruence_5']) +
        decay_weights['10'] * (df['sign_congruence_10'] * df['mag_congruence_10']) +
        decay_weights['20'] * (df['sign_congruence_20'] * df['mag_congruence_20'])
    ) / sum(decay_weights.values())
    
    # Volume acceleration contribution
    volume_accel_contribution = df['volume_accel'] * np.sign(df['weighted_price_mom'])
    
    # Divergence penalty/bonus
    directional_alignment = np.sign(df['weighted_price_mom']) * np.sign(df['weighted_volume_mom'])
    divergence_strength = np.abs(df['weighted_price_mom'] - df['weighted_volume_mom'])
    divergence_score = directional_alignment * divergence_strength * df['timeframe_consistency']
    
    # Combine components
    raw_factor = (
        base_congruence * 0.6 +
        volume_accel_contribution * 0.2 +
        divergence_score * 0.2
    ) * df['weighted_price_mom']
    
    # Regime adjustment (reduce sensitivity in high volatility)
    regime_adjustment = 1.0 / (1.0 + df['regime_score'])
    adjusted_factor = raw_factor * regime_adjustment
    
    # Apply exponential smoothing for robustness
    final_factor = adjusted_factor.ewm(span=3, adjust=False).mean()
    
    return final_factor
