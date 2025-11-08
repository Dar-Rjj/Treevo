import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining volatility regime classification, multi-timeframe momentum convergence,
    cross-sectional relative positioning, and volume-price divergence with volatility scaling.
    """
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic price and volume features
    close = df['close']
    high = df['high']
    low = df['low']
    volume = df['volume']
    
    # Volatility regime classification
    current_range = (high - low) / close
    avg_range_5d = current_range.rolling(window=5).mean()
    avg_range_20d = current_range.rolling(window=20).mean()
    
    returns = close.pct_change()
    vol_5d = returns.rolling(window=5).std()
    vol_20d = returns.rolling(window=20).std()
    
    # Volatility regime thresholds
    high_vol_threshold = 1.5 * avg_range_20d
    low_vol_threshold = 0.7 * avg_range_20d
    
    # Multi-timeframe momentum
    mom_3d_price = (close - close.shift(3)) / close.shift(3)
    mom_10d_price = (close - close.shift(10)) / close.shift(10)
    mom_20d_price = (close - close.shift(20)) / close.shift(20)
    
    mom_3d_vol = volume / volume.shift(3) - 1
    mom_10d_vol = volume / volume.shift(10) - 1
    mom_20d_vol = volume / volume.shift(20) - 1
    
    # Momentum convergence scoring
    direction_alignment = np.sign(mom_3d_price) * np.sign(mom_10d_price) * np.sign(mom_20d_price)
    volume_price_alignment = (mom_3d_price * mom_3d_vol) + (mom_10d_price * mom_10d_vol) + (mom_20d_price * mom_20d_vol)
    multi_timeframe_strength = abs(mom_3d_price) + abs(mom_10d_price) + abs(mom_20d_price)
    
    # Volume-price divergence detection
    price_div_short_med = mom_3d_price - mom_10d_price
    volume_div_short_med = mom_3d_vol - mom_10d_vol
    combined_div_short_med = price_div_short_med * volume_div_short_med
    
    price_div_med_long = mom_10d_price - mom_20d_price
    volume_div_med_long = mom_10d_vol - mom_20d_vol
    combined_div_med_long = price_div_med_long * volume_div_med_long
    
    # Volatility-adjusted divergence
    volatility_scaled_div_short_med = combined_div_short_med / (current_range + 1e-8)
    volatility_scaled_div_med_long = combined_div_med_long / (current_range + 1e-8)
    
    # Intraday patterns
    gap_magnitude = (df['open'] - close.shift(1)) / close.shift(1)
    gap_to_range_ratio = abs(gap_magnitude) / (current_range + 1e-8)
    
    morning_strength = (high - df['open']) / df['open']
    closing_deviation = (close - (high + low)/2) / ((high + low)/2 + 1e-8)
    
    # Cross-sectional rankings (using rolling percentiles)
    range_rank = current_range.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    momentum_rank_3d = mom_3d_price.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    divergence_rank = volatility_scaled_div_short_med.rolling(window=50, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    
    # Regime-adaptive signal integration
    for i in range(len(df)):
        if i < 20:  # Ensure enough data for calculations
            result.iloc[i] = 0
            continue
            
        # Volatility regime classification
        if current_range.iloc[i] > high_vol_threshold.iloc[i]:
            regime = 'high'
        elif current_range.iloc[i] < low_vol_threshold.iloc[i]:
            regime = 'low'
        else:
            regime = 'normal'
        
        # Regime-specific weighting
        if regime == 'high':
            # Emphasize volume confirmation and volatility breakouts
            volume_weight = 0.6
            momentum_weight = 0.2
            divergence_weight = 0.2
            regime_multiplier = 1.2
        elif regime == 'low':
            # Emphasize momentum convergence and subtle divergences
            volume_weight = 0.3
            momentum_weight = 0.5
            divergence_weight = 0.2
            regime_multiplier = 1.0
        else:  # normal
            volume_weight = 0.4
            momentum_weight = 0.4
            divergence_weight = 0.2
            regime_multiplier = 1.0
        
        # Calculate regime-adaptive factor components
        volume_component = (
            volume_price_alignment.iloc[i] * 0.4 +
            (mom_3d_vol.iloc[i] + mom_10d_vol.iloc[i] + mom_20d_vol.iloc[i]) * 0.3 +
            gap_to_range_ratio.iloc[i] * 0.3
        )
        
        momentum_component = (
            direction_alignment.iloc[i] * multi_timeframe_strength.iloc[i] * 0.4 +
            morning_strength.iloc[i] * 0.3 +
            momentum_rank_3d.iloc[i] * 0.3
        )
        
        divergence_component = (
            volatility_scaled_div_short_med.iloc[i] * 0.5 +
            volatility_scaled_div_med_long.iloc[i] * 0.3 +
            divergence_rank.iloc[i] * 0.2
        )
        
        # Combine components with regime-adaptive weights
        factor_value = (
            volume_component * volume_weight +
            momentum_component * momentum_weight +
            divergence_component * divergence_weight
        ) * regime_multiplier
        
        # Apply cross-sectional adjustment
        cross_sectional_adjustment = (
            range_rank.iloc[i] * 0.3 +
            momentum_rank_3d.iloc[i] * 0.4 +
            divergence_rank.iloc[i] * 0.3
        )
        
        result.iloc[i] = factor_value * cross_sectional_adjustment
    
    # Normalize the factor
    result = (result - result.rolling(window=50, min_periods=20).mean()) / (result.rolling(window=50, min_periods=20).std() + 1e-8)
    
    return result
