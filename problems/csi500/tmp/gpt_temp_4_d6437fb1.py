import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Volume-Weighted Multi-Timeframe Momentum
    # Short-Term Momentum Component
    mom_1d = (df['close'] - df['close'].shift(1)) / df['close'].shift(1)
    mom_3d = (df['close'] - df['close'].shift(3)) / df['close'].shift(3)
    mom_5d = (df['close'] - df['close'].shift(5)) / df['close'].shift(5)
    
    # Volume Acceleration Component
    vol_change_1d = df['volume'] / df['volume'].shift(1) - 1
    vol_change_3d = df['volume'] / df['volume'].shift(3) - 1
    vol_change_5d = df['volume'] / df['volume'].shift(5) - 1
    
    # Combined Signals
    short_term_weighted = (mom_1d * vol_change_1d).shift(1)
    medium_term_weighted = (mom_3d * vol_change_3d).shift(2)
    long_term_weighted = (mom_5d * vol_change_5d).shift(3)
    
    # Gap Momentum with Volume Persistence
    # Gap Analysis Component
    overnight_gap = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
    intraday_mom = (df['close'] - df['open']) / df['open']
    gap_momentum_interaction = overnight_gap * intraday_mom
    
    # Volume Persistence Component
    vol_above_3d_avg = df['volume'] > (df['volume'].shift(1) + df['volume'].shift(2) + df['volume'].shift(3)) / 3
    vol_above_5d_avg = df['volume'] > (df['volume'].shift(1) + df['volume'].shift(2) + df['volume'].shift(3) + df['volume'].shift(4) + df['volume'].shift(5)) / 5
    vol_persistence_score = vol_above_3d_avg.astype(int) + vol_above_5d_avg.astype(int)
    
    # Enhanced Gap Signals
    lagged_gap_momentum = gap_momentum_interaction.shift(1) * vol_persistence_score
    multi_period_gap_momentum = gap_momentum_interaction.rolling(window=3).mean() * vol_persistence_score
    
    # Range-Based Volume Alignment
    # Price Range Component
    normalized_range = (df['high'] - df['low']) / ((df['high'] + df['low']) / 2)
    range_momentum = normalized_range / normalized_range.shift(1) - 1
    range_expansion = normalized_range - normalized_range.shift(3)
    
    # Volume-Range Alignment
    def rolling_corr_3d(x, y):
        return x.rolling(window=3).corr(y)
    
    def rolling_corr_5d(x, y):
        return x.rolling(window=5).corr(y)
    
    vol_range_corr_3d = rolling_corr_3d(df['volume'], normalized_range)
    vol_range_corr_5d = rolling_corr_5d(df['volume'], normalized_range)
    alignment_strength = (vol_range_corr_3d + vol_range_corr_5d) / 2
    
    # Range-Volume Factors
    lagged_range_momentum_aligned = range_momentum.shift(1) * alignment_strength
    range_expansion_volume_confirmed = range_expansion * alignment_strength
    
    # Multi-Timeframe Price-Volume Consistency
    # Price Consistency Measures
    price_direction = np.sign(df['close'].diff())
    price_consistency_3d = price_direction.rolling(window=3).apply(lambda x: len(set(x)) == 1, raw=False).fillna(0)
    price_consistency_5d = price_direction.rolling(window=5).apply(lambda x: len(set(x)) == 1, raw=False).fillna(0)
    price_trend_strength = (price_consistency_3d + price_consistency_5d) / 2
    
    # Volume Consistency Measures
    volume_direction = np.sign(df['volume'].diff())
    volume_consistency_3d = volume_direction.rolling(window=3).apply(lambda x: len(set(x)) == 1, raw=False).fillna(0)
    volume_consistency_5d = volume_direction.rolling(window=5).apply(lambda x: len(set(x)) == 1, raw=False).fillna(0)
    volume_trend_strength = (volume_consistency_3d + volume_consistency_5d) / 2
    
    # Consistency Alignment Factors
    price_volume_consistency_match = (price_trend_strength * volume_trend_strength).shift(2)
    consistency_divergence = abs(price_trend_strength - volume_trend_strength).shift(1)
    
    # Combine all factors with equal weights
    factor = (
        short_term_weighted + medium_term_weighted + long_term_weighted +
        lagged_gap_momentum + multi_period_gap_momentum +
        lagged_range_momentum_aligned + range_expansion_volume_confirmed +
        price_volume_consistency_match - consistency_divergence
    )
    
    return factor
