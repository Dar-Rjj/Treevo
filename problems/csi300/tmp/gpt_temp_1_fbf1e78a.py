import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    alpha = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price changes
    df['price_change'] = df['close'] - df['close'].shift(1)
    df['abs_price_change'] = abs(df['price_change'])
    df['range'] = df['high'] - df['low']
    
    # Calculate rolling windows for various timeframes
    for window in [5, 13, 34]:
        df[f'volume_sum_{window}'] = df['volume'].rolling(window=window, min_periods=1).sum()
        df[f'range_sum_{window}'] = df['range'].rolling(window=window, min_periods=1).sum()
        df[f'abs_change_sum_{window}'] = df['abs_price_change'].rolling(window=window, min_periods=1).sum()
    
    # Directional Volume Asymmetry
    up_volume = pd.Series(0, index=df.index)
    down_volume = pd.Series(0, index=df.index)
    down_abs_change = pd.Series(0, index=df.index)
    total_abs_change = pd.Series(0, index=df.index)
    
    for i in range(5):
        shifted_close = df['close'].shift(i)
        prev_close = df['close'].shift(i+1)
        shifted_volume = df['volume'].shift(i)
        
        up_mask = (shifted_close > prev_close) & (prev_close.notna())
        down_mask = (shifted_close < prev_close) & (prev_close.notna())
        
        up_volume += np.where(up_mask, shifted_volume, 0)
        down_volume += np.where(down_mask, shifted_volume, 0)
        down_abs_change += np.where(down_mask, abs(shifted_close - prev_close), 0)
        total_abs_change += abs(shifted_close - prev_close)
    
    # Avoid division by zero
    up_volume_ratio = np.where(down_volume > 0, up_volume / down_volume, 1)
    down_day_concentration = np.where(total_abs_change > 0, down_abs_change / total_abs_change, 0)
    
    # Volume-Price Divergence
    current_direction = np.sign(df['price_change'])
    volume_price_divergence = (up_volume_ratio - 1) * current_direction
    
    # Multi-Timeframe Asymmetry Patterns
    df['vp_divergence_5'] = volume_price_divergence.rolling(window=5, min_periods=1).mean()
    df['vp_divergence_13'] = volume_price_divergence.rolling(window=13, min_periods=1).mean()
    df['vp_divergence_34'] = volume_price_divergence.rolling(window=34, min_periods=1).mean()
    
    # Asymmetry Persistence
    vp_sign = np.sign(volume_price_divergence)
    directional_consistency = vp_sign.rolling(window=5, min_periods=1).apply(
        lambda x: np.mean(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    
    magnitude_trend = volume_price_divergence / volume_price_divergence.shift(4).replace(0, 1)
    
    vp_rolling_std = volume_price_divergence.rolling(window=5, min_periods=1).std()
    vp_rolling_mean = volume_price_divergence.rolling(window=5, min_periods=1).mean()
    asymmetry_stability = 1 - (vp_rolling_std / abs(vp_rolling_mean).replace(0, 1))
    
    # Volatility-Regime Classification
    volatility_ratio = df['range'] / df['close']
    high_vol = volatility_ratio > 0.03
    medium_vol = (volatility_ratio >= 0.01) & (volatility_ratio <= 0.03)
    low_vol = volatility_ratio < 0.01
    
    # Regime-Specific Momentum Components
    range_sum_5 = df['range'].rolling(window=5, min_periods=1).sum()
    abs_change_sum_8 = df['abs_price_change'].rolling(window=8, min_periods=1).sum()
    
    high_vol_momentum = (df['close'] - df['close'].shift(5)) / range_sum_5.replace(0, 1)
    medium_vol_momentum = (df['close'] - df['close'].shift(8)) / abs_change_sum_8.replace(0, 1)
    
    high_12 = df['high'].rolling(window=13, min_periods=1).max()
    low_12 = df['low'].rolling(window=13, min_periods=1).min()
    low_vol_momentum = (df['close'] - df['close'].shift(13)) / (high_12 - low_12).replace(0, 1)
    
    regime_momentum = pd.Series(0.0, index=df.index)
    regime_momentum[high_vol] = high_vol_momentum[high_vol]
    regime_momentum[medium_vol] = medium_vol_momentum[medium_vol]
    regime_momentum[low_vol] = low_vol_momentum[low_vol]
    
    # Price-Efficiency Asymmetry Analysis
    df['opening_efficiency'] = (df['open'] - df['low']) / df['range'].replace(0, 1)
    df['closing_efficiency'] = (df['close'] - df['low']) / df['range'].replace(0, 1)
    efficiency_gap = df['closing_efficiency'] - df['opening_efficiency']
    
    # Multi-Day Efficiency Asymmetry
    up_efficiency = pd.Series(0.0, index=df.index)
    down_efficiency = pd.Series(0.0, index=df.index)
    up_count = pd.Series(0, index=df.index)
    down_count = pd.Series(0, index=df.index)
    
    for i in range(5):
        shifted_close = df['close'].shift(i)
        prev_close = df['close'].shift(i+1)
        shifted_eff = df['closing_efficiency'].shift(i)
        
        up_mask = (shifted_close > prev_close) & (prev_close.notna())
        down_mask = (shifted_close < prev_close) & (prev_close.notna())
        
        up_efficiency += np.where(up_mask, shifted_eff, 0)
        down_efficiency += np.where(down_mask, shifted_eff, 0)
        up_count += up_mask.astype(int)
        down_count += down_mask.astype(int)
    
    up_day_efficiency = np.where(up_count > 0, up_efficiency / up_count, 0)
    down_day_efficiency = np.where(down_count > 0, down_efficiency / down_count, 0)
    efficiency_bias = up_day_efficiency - down_day_efficiency
    
    # Asymmetric Breakout Confirmation
    high_5 = df['high'].rolling(window=5, min_periods=1).max().shift(1)
    low_5 = df['low'].rolling(window=5, min_periods=1).min().shift(1)
    volume_mean_5 = df['volume'].rolling(window=5, min_periods=1).mean().shift(1)
    
    up_breakout = (df['close'] > high_5) & (df['volume'] > volume_mean_5)
    down_breakout = (df['close'] < low_5) & (df['volume'] > volume_mean_5)
    breakout_asymmetry = up_breakout.astype(int) - down_breakout.astype(int)
    
    breakout_strength = abs(df['price_change']) / df['range'].replace(0, 1)
    volume_confirmation = df['volume'] / volume_mean_5.replace(0, 1)
    efficiency_alignment = df['closing_efficiency'] * np.sign(df['price_change'])
    
    # Multi-Timeframe Breakout Consistency
    breakout_asymmetry_3 = breakout_asymmetry.rolling(window=3, min_periods=1).mean()
    breakout_asymmetry_8 = breakout_asymmetry.rolling(window=8, min_periods=1).mean()
    breakout_asymmetry_21 = breakout_asymmetry.rolling(window=21, min_periods=1).mean()
    
    # Adaptive Signal Integration
    volume_price_alignment = np.sign(df['vp_divergence_5']) * np.sign(regime_momentum)
    efficiency_momentum_alignment = np.sign(efficiency_bias) * np.sign(regime_momentum)
    breakout_momentum_confirmation = np.sign(breakout_asymmetry) * np.sign(regime_momentum)
    
    # Regime-Adaptive Weighting
    vp_weight = pd.Series(0.5, index=df.index)
    eff_weight = pd.Series(0.5, index=df.index)
    
    vp_weight[high_vol] = 0.6
    eff_weight[high_vol] = 0.4
    vp_weight[low_vol] = 0.4
    eff_weight[low_vol] = 0.6
    
    # Core Asymmetry Components
    primary_asymmetry = df['vp_divergence_5'] * efficiency_bias
    momentum_alignment_factor = regime_momentum * volume_price_alignment
    breakout_confirmation_factor = breakout_asymmetry * breakout_strength
    
    # Regime-Weighted Combination
    regime_weighted_factor = (vp_weight * primary_asymmetry + 
                             eff_weight * momentum_alignment_factor + 
                             0.3 * breakout_confirmation_factor)
    
    # Transition Enhancement
    volatility_change = volatility_ratio - volatility_ratio.shift(5)
    transition_strength = abs(volatility_change) * abs(regime_momentum)
    transition_boost = regime_weighted_factor * (1 + abs(transition_strength))
    
    # Consistency Filter
    consistency_multiplier = regime_weighted_factor * asymmetry_stability
    
    # Final Alpha Construction
    final_alpha = 0.6 * regime_weighted_factor + 0.3 * transition_boost + 0.1 * consistency_multiplier
    
    # Normalize and clean
    alpha = final_alpha.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return alpha
