import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor using multi-timeframe price-volume divergence with adaptive volatility regime,
    volume anomaly detection, intraday efficiency, price context, and trend persistence.
    """
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Multi-Timeframe Price-Volume Divergence
    timeframes = [3, 7, 21, 63]
    
    # Calculate divergences for each timeframe
    divergences = {}
    for tf in timeframes:
        # Price momentum
        price_momentum = df['close'] / df['close'].shift(tf)
        
        # Volume momentum
        volume_momentum = df['volume'] / df['volume'].shift(tf)
        
        # Divergence
        divergences[tf] = price_momentum - volume_momentum
    
    # Adaptive Volatility Regime Framework
    # Calculate log returns
    log_returns = np.log(df['close'] / df['close'].shift(1))
    
    # Volatility measurements
    vol_ultra_short = log_returns.rolling(window=3).std()
    vol_short = log_returns.rolling(window=7).std()
    vol_medium = log_returns.rolling(window=21).std()
    vol_long = log_returns.rolling(window=63).std()
    
    # Initialize weights
    weights = pd.DataFrame(index=df.index, columns=['ultra_short', 'short', 'medium', 'long'])
    
    # Regime-based weight allocation
    for i in df.index:
        if vol_ultra_short.loc[i] > 2.0 * vol_medium.loc[i]:
            # High Volatility
            weights.loc[i] = [0.6, 0.3, 0.1, 0.0]
        elif vol_ultra_short.loc[i] > 1.5 * vol_medium.loc[i]:
            # Elevated Volatility
            weights.loc[i] = [0.4, 0.4, 0.2, 0.0]
        elif vol_ultra_short.loc[i] < 0.5 * vol_medium.loc[i]:
            # Low Volatility
            weights.loc[i] = [0.1, 0.2, 0.3, 0.4]
        else:
            # Normal Volatility
            weights.loc[i] = [0.2, 0.3, 0.3, 0.2]
    
    # Robust Volume Anomaly Detection
    volume_median = df['volume'].rolling(window=21).median()
    volume_q25 = df['volume'].rolling(window=21).quantile(0.25)
    volume_q75 = df['volume'].rolling(window=21).quantile(0.75)
    volume_iqr = volume_q75 - volume_q25
    volume_zscore = (df['volume'] - volume_median) / volume_iqr
    
    # Anomaly classification multiplier
    volume_multiplier = pd.Series(1.0, index=df.index)
    volume_multiplier[volume_zscore > 3.0] = 2.5
    volume_multiplier[(volume_zscore > 2.0) & (volume_zscore <= 3.0)] = 2.0
    volume_multiplier[(volume_zscore > 1.0) & (volume_zscore <= 2.0)] = 1.5
    
    # Intraday Market Microstructure Analysis
    daily_range_ratio = (df['high'] - df['low']) / df['close']
    close_to_open_efficiency = abs(df['close'] - df['open']) / (df['high'] - df['low'])
    
    # Efficiency scoring
    efficiency_score = pd.Series(0.8, index=df.index)
    efficiency_score[(close_to_open_efficiency < 0.6) & (daily_range_ratio > 0.01)] = 1.0
    efficiency_score[(close_to_open_efficiency < 0.4) & (daily_range_ratio > 0.015)] = 1.2
    efficiency_score[(close_to_open_efficiency < 0.2) & (daily_range_ratio > 0.02)] = 1.4
    
    # Multi-Level Price Context Framework
    high_7d = df['high'].rolling(window=7).max()
    low_7d = df['low'].rolling(window=7).min()
    high_21d = df['high'].rolling(window=21).max()
    low_21d = df['low'].rolling(window=21).min()
    
    # Position-based adjustment
    position_multiplier = pd.Series(1.0, index=df.index)
    position_multiplier[df['close'] > 0.98 * high_7d] = 0.7
    position_multiplier[df['close'] > 0.95 * high_21d] = 0.85
    position_multiplier[df['close'] < 1.02 * low_7d] = 1.3
    position_multiplier[df['close'] < 1.05 * low_21d] = 1.15
    
    # Trend Persistence Assessment
    ultra_short_trend = np.sign(df['close'] - df['close'].shift(3))
    short_trend = np.sign(df['close'] - df['close'].shift(7))
    medium_trend = np.sign(df['close'] - df['close'].shift(21))
    long_trend = np.sign(df['close'] - df['close'].shift(63))
    
    # Trend consistency score
    trend_alignment = pd.DataFrame({
        'ultra_short': ultra_short_trend,
        'short': short_trend,
        'medium': medium_trend,
        'long': long_trend
    })
    
    same_direction_count = trend_alignment.apply(
        lambda x: sum(x == x.iloc[0]) if x.iloc[0] != 0 else 0, axis=1
    )
    
    trend_consistency_score = pd.Series(0.9, index=df.index)
    trend_consistency_score[same_direction_count == 2] = 1.0
    trend_consistency_score[same_direction_count == 3] = 1.15
    trend_consistency_score[same_direction_count == 4] = 1.3
    
    # Hierarchical Alpha Construction
    for i in df.index:
        if pd.isna(vol_medium.loc[i]) or pd.isna(divergences[3].loc[i]):
            continue
            
        # Base Divergence
        base_divergence = (
            divergences[3].loc[i] * weights.loc[i, 'ultra_short'] +
            divergences[7].loc[i] * weights.loc[i, 'short'] +
            divergences[21].loc[i] * weights.loc[i, 'medium'] +
            divergences[63].loc[i] * weights.loc[i, 'long']
        )
        
        # Volume Adjusted
        volume_adjusted = base_divergence * volume_multiplier.loc[i]
        
        # Efficiency Enhanced
        efficiency_enhanced = volume_adjusted * efficiency_score.loc[i]
        
        # Context Refined
        context_refined = efficiency_enhanced * position_multiplier.loc[i]
        
        # Final Alpha
        final_alpha = context_refined * trend_consistency_score.loc[i]
        
        result.loc[i] = final_alpha
    
    return result
