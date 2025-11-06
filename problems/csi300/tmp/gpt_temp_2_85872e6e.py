import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price Gap Momentum Divergence factor combining overnight momentum strength,
    gap filling efficiency, volume-gap interaction, and multi-timeframe alignment.
    """
    data = df.copy()
    
    # Calculate basic price metrics
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_return'] = (data['close'] - data['open']) / data['open']
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    data['prev_range'] = data['daily_range'].shift(1)
    
    # Overnight vs Intraday Momentum
    ## Overnight Return Patterns
    data['gap_direction'] = np.sign(data['overnight_gap'])
    
    # Consecutive overnight gaps in same direction (3,5,8 days)
    for window in [3, 5, 8]:
        data[f'consecutive_gaps_{window}'] = (
            data['gap_direction'].rolling(window=window).apply(
                lambda x: len(x) if len(set(x)) == 1 and x.iloc[0] != 0 else 0, raw=False
            )
        )
    
    # Overnight gap magnitude vs previous day range
    data['gap_vs_range'] = abs(data['overnight_gap']) / (data['prev_range'] + 1e-8)
    
    ## Intraday Momentum Persistence
    # Open-to-close return consistency (5,8,13 days)
    for window in [5, 8, 13]:
        data[f'intraday_consistency_{window}'] = (
            data['intraday_return'].rolling(window=window).apply(
                lambda x: (x * x.shift(1)).sum() if len(x) == window else 0, raw=False
            )
        )
    
    # Intraday trend reversal frequency
    data['intraday_reversal'] = (
        (data['intraday_return'] * data['intraday_return'].shift(1) < 0).rolling(window=10).mean()
    )
    
    # Gap Filling Behavior
    ## Gap Closure Speed
    # Calculate gap filling progress
    data['gap_fill_progress'] = np.where(
        data['overnight_gap'] > 0,
        (data['low'] - data['open']) / (data['close'].shift(1) - data['open'] + 1e-8),
        (data['high'] - data['open']) / (data['close'].shift(1) - data['open'] + 1e-8)
    )
    
    # Time to fill overnight gaps (using rolling window)
    data['gap_fill_speed'] = data['gap_fill_progress'].rolling(window=3).mean()
    
    ## Gap Exhaustion Signals
    # Failed gap attempts (gap doesn't sustain direction)
    data['gap_failure'] = (
        (data['overnight_gap'] * data['intraday_return'] < 0) & 
        (abs(data['intraday_return']) > abs(data['overnight_gap']))
    ).astype(int)
    
    # Volume-Gap Interaction
    ## Volume Concentration Around Gaps
    # Pre-gap volume buildup
    data['pre_gap_volume'] = data['volume'].shift(1) / data['volume'].rolling(window=5).mean().shift(1)
    
    # Post-gap volume distribution
    data['post_gap_volume_ratio'] = data['volume'] / data['volume'].rolling(window=3).mean()
    
    ## Gap Volume Efficiency
    # Gap size relative to preceding volume
    data['gap_volume_efficiency'] = (
        abs(data['overnight_gap']) / (data['volume'].shift(1) / data['volume'].rolling(window=10).mean().shift(1) + 1e-8)
    )
    
    # Volume-weighted gap persistence
    data['volume_weighted_gap'] = data['overnight_gap'] * data['pre_gap_volume']
    
    # Multi-Timeframe Gap Alignment
    ## Short-term vs Medium-term Gap Consistency
    # Directional alignment of gaps across periods
    short_term_gap = data['overnight_gap'].rolling(window=3).mean()
    medium_term_gap = data['overnight_gap'].rolling(window=8).mean()
    data['gap_alignment'] = np.sign(short_term_gap) * np.sign(medium_term_gap)
    
    # Gap magnitude correlation across timeframes
    data['gap_magnitude_correlation'] = (
        data['overnight_gap'].rolling(window=5).std() / 
        (data['overnight_gap'].rolling(window=10).std() + 1e-8)
    )
    
    ## Gap Cluster Analysis
    # Density of gaps in recent period
    data['gap_density'] = (
        (abs(data['overnight_gap']) > data['overnight_gap'].rolling(window=10).std()).rolling(window=5).sum()
    )
    
    # Gap clustering by size and direction
    large_gaps = abs(data['overnight_gap']) > data['overnight_gap'].rolling(window=20).std()
    same_direction = data['gap_direction'] == data['gap_direction'].shift(1)
    data['gap_clustering'] = (large_gaps & same_direction).rolling(window=5).sum()
    
    # Final Alpha Factor Construction
    
    # 1. Overnight momentum strength component
    overnight_strength = (
        data['consecutive_gaps_5'] * data['gap_vs_range'] * 
        np.sign(data['overnight_gap']) * (1 - data['intraday_reversal'])
    )
    
    # 2. Gap filling efficiency component
    gap_filling_efficiency = (
        data['gap_fill_speed'] * (1 - data['gap_failure']) * 
        np.where(data['gap_fill_progress'] > 0.5, 1, -1)
    )
    
    # 3. Volume-gap interaction score
    volume_gap_score = (
        data['gap_volume_efficiency'] * data['volume_weighted_gap'] * 
        data['post_gap_volume_ratio']
    )
    
    # 4. Multi-timeframe gap alignment adjustment
    timeframe_alignment = (
        data['gap_alignment'] * data['gap_magnitude_correlation'] * 
        (1 + data['gap_density'] / 5) * (1 + data['gap_clustering'] / 3)
    )
    
    # Combine all components
    alpha_factor = (
        overnight_strength * 
        gap_filling_efficiency * 
        volume_gap_score * 
        timeframe_alignment
    )
    
    # Normalize and clean
    alpha_factor = alpha_factor.replace([np.inf, -np.inf], np.nan)
    alpha_factor = alpha_factor.fillna(0)
    
    return alpha_factor
