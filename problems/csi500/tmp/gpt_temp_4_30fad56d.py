import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(data):
    # Volatility-Regime Adjusted Momentum
    # Dual-Timeframe Returns
    short_return = data['close'] / data['close'].shift(5) - 1
    medium_return = data['close'] / data['close'].shift(20) - 1
    
    # Estimate Volatility Regimes
    tr = pd.DataFrame({
        'hl': data['high'] - data['low'],
        'hcp': abs(data['high'] - data['close'].shift(1)),
        'lcp': abs(data['low'] - data['close'].shift(1))
    }).max(axis=1)
    tr_avg = tr.rolling(20).mean()
    
    high_vol_threshold = tr_avg.rolling(60).quantile(0.8)
    low_vol_threshold = tr_avg.rolling(60).quantile(0.2)
    
    high_vol_regime = tr_avg > high_vol_threshold
    low_vol_regime = tr_avg < low_vol_threshold
    
    # Apply Regime-Specific Weighting
    momentum_factor = pd.Series(index=data.index, dtype=float)
    momentum_factor[high_vol_regime] = 0.7 * medium_return[high_vol_regime] + 0.3 * short_return[high_vol_regime]
    momentum_factor[low_vol_regime] = 0.3 * medium_return[low_vol_regime] + 0.7 * short_return[low_vol_regime]
    momentum_factor[~high_vol_regime & ~low_vol_regime] = 0.5 * medium_return[~high_vol_regime & ~low_vol_regime] + 0.5 * short_return[~high_vol_regime & ~low_vol_regime]
    
    # Volume-Trend Confirmed Price Efficiency
    # Price Efficiency Ratio
    daily_range = (data['high'] - data['low']) / data['close'].shift(1)
    abs_price_change = abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    efficiency = abs_price_change / daily_range.replace(0, np.nan)
    
    # Volume Trend Strength
    volume_momentum = data['volume'] / data['volume'].shift(5) - 1
    volume_changes = data['volume'].pct_change()
    volume_consistency = volume_changes.rolling(5).apply(lambda x: sum(np.sign(x) == np.sign(x.iloc[-1])) if len(x) == 5 else np.nan, raw=False)
    volume_trend_score = volume_momentum * volume_consistency
    
    # Combine Efficiency with Volume Confirmation
    efficiency_factor = efficiency * volume_trend_score
    
    # Multi-Timeframe Breakout Consistency
    # Identify Breakout Levels
    short_resistance = data['high'].rolling(3).max().shift(1)
    medium_resistance = data['high'].rolling(10).max().shift(1)
    short_support = data['low'].rolling(3).min().shift(1)
    medium_support = data['low'].rolling(10).min().shift(1)
    
    # Assess Breakout Strength
    short_breakout = data['close'] > short_resistance
    medium_breakout = data['close'] > medium_resistance
    breakout_magnitude = (data['close'] - short_resistance) / short_resistance
    volume_confirmation = data['volume'] > data['volume'].shift(1)
    
    # Score Breakout Consistency
    timeframe_alignment = (short_breakout.astype(int) + medium_breakout.astype(int)) / 2
    volume_multiplier = volume_confirmation.astype(int) * 1.5 + (~volume_confirmation).astype(int) * 0.5
    breakout_factor = breakout_magnitude * timeframe_alignment * volume_multiplier
    
    # Amount-Volume Divergence Signal
    # Calculate Trading Intensity
    amount_per_volume = data['amount'] / data['volume'].replace(0, np.nan)
    av_5ma = amount_per_volume.rolling(5).mean()
    intensity_deviation = amount_per_volume / av_5ma - 1
    
    # Analyze Price-Intensity Relationship
    price_trend = data['close'] / data['close'].shift(5) - 1
    intensity_trend = amount_per_volume / amount_per_volume.shift(5) - 1
    
    # Generate Divergence Signals
    bullish_divergence = (price_trend < 0) & (intensity_trend > 0)
    bearish_divergence = (price_trend > 0) & (intensity_trend < 0)
    divergence_direction = bullish_divergence.astype(int) - bearish_divergence.astype(int)
    
    divergence_factor = intensity_deviation * divergence_direction
    
    # Combine all factors with equal weighting
    final_factor = (momentum_factor + efficiency_factor + breakout_factor + divergence_factor) / 4
    
    return final_factor
