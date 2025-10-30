import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volatility-Scaled Multi-Timeframe Momentum
    # Short-term momentum (3-day)
    st_momentum = df['close'] / df['close'].shift(3) - 1
    st_weight = 0.7
    
    # Medium-term momentum (10-day)
    mt_momentum = df['close'] / df['close'].shift(10) - 1
    mt_weight = 0.3
    
    # Momentum combination
    combined_momentum = st_weight * st_momentum + mt_weight * mt_momentum
    
    # Dynamic volatility scaling
    volatility_proxy = (df['high'] - df['low']).rolling(window=5).mean()
    volatility_scaled_momentum = combined_momentum / volatility_proxy
    momentum_factor = volatility_scaled_momentum.rolling(window=3).mean()
    
    # Regime-Dependent Price Reversal
    # Price extremity detection
    high_5d = df['high'].rolling(window=5).max()
    low_5d = df['low'].rolling(window=5).min()
    position_in_range = (df['close'] - low_5d) / (high_5d - low_5d)
    
    upper_extreme = position_in_range > 0.8
    lower_extreme = position_in_range < 0.2
    
    # Volume acceleration analysis
    volume_momentum = df['volume'] / df['volume'].shift(1)
    avg_volume_5d = df['volume'].rolling(window=5).mean()
    volume_vs_avg = df['volume'] / avg_volume_5d
    volume_spike = (volume_momentum > 1.2) & (volume_vs_avg > 1.2)
    
    # Regime-weighted reversal signals
    bullish_reversal = lower_extreme & volume_spike
    bearish_reversal = upper_extreme & volume_spike
    
    # Dynamic weighting based on recent signal frequency
    bullish_freq = bullish_reversal.rolling(window=10).mean()
    bearish_freq = bearish_reversal.rolling(window=10).mean()
    
    reversal_signal = bullish_reversal.astype(float) - bearish_reversal.astype(float)
    reversal_factor = reversal_signal * (1 - (bullish_freq + bearish_freq))
    
    # Simplified Volume-Price Alignment
    # Recent price-volume correlation
    price_returns_3d = df['close'].pct_change(periods=3)
    volume_changes_3d = df['volume'].pct_change(periods=3)
    price_volume_corr = price_returns_3d.rolling(window=3).corr(volume_changes_3d)
    
    # Multi-timeframe volume ratios
    avg_volume_3d = df['volume'].rolling(window=3).mean()
    avg_volume_10d = df['volume'].rolling(window=10).mean()
    
    volume_ratio_st = df['volume'] / avg_volume_3d
    volume_ratio_mt = avg_volume_3d / avg_volume_10d
    volume_ratio_combined = volume_ratio_st + volume_ratio_mt
    
    # Clear reversal alignment
    price_up = df['close'] > df['close'].shift(1)
    volume_up = df['volume'] > df['volume'].shift(1)
    positive_alignment = price_up & volume_up
    negative_alignment = (~price_up) & (~volume_up)
    
    alignment_factor = positive_alignment.astype(float) - negative_alignment.astype(float)
    volume_price_factor = price_volume_corr * volume_ratio_combined * alignment_factor
    
    # Dynamic Timeframe Liquidity Quality
    # Spread-volume dynamics
    daily_spread = df['high'] - df['low']
    volume_weighted_spread = daily_spread * df['volume']
    spread_trend = volume_weighted_spread.rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    
    # Liquidity regime classification
    volume_trend = df['volume'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    
    improving_liquidity = (spread_trend < 0) & (volume_trend > 0)
    deteriorating_liquidity = (spread_trend > 0) & (volume_trend < 0)
    
    # Timeframe adaptation
    liquidity_short_term = improving_liquidity.astype(float) - deteriorating_liquidity.astype(float)
    liquidity_medium_term = liquidity_short_term.rolling(window=10).mean()
    
    liquidity_factor = liquidity_short_term * 0.6 + liquidity_medium_term * 0.4
    
    # Combine all factors with equal weights
    final_factor = (
        momentum_factor.fillna(0) * 0.25 +
        reversal_factor.fillna(0) * 0.25 +
        volume_price_factor.fillna(0) * 0.25 +
        liquidity_factor.fillna(0) * 0.25
    )
    
    return final_factor
