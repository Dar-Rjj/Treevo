import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate basic technical indicators
    df['prev_close'] = df['close'].shift(1)
    df['true_range'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['prev_close']),
            abs(df['low'] - df['prev_close'])
        )
    )
    df['atr_5'] = df['true_range'].rolling(window=5).mean()
    df['atr_20'] = df['true_range'].rolling(window=20).mean()
    df['return_5'] = df['close'].pct_change(5)
    df['return_3'] = df['close'].pct_change(3)
    df['return_6'] = df['close'].pct_change(6)
    
    # Volatility Regime Momentum
    high_vol_threshold = df['atr_20'].rolling(window=20).quantile(0.7)
    low_vol_threshold = df['atr_20'].rolling(window=20).quantile(0.3)
    
    high_vol_mask = df['atr_5'] > high_vol_threshold
    low_vol_mask = df['atr_5'] < low_vol_threshold
    
    high_vol_momentum = df['return_5'].where(high_vol_mask, 0)
    low_vol_momentum = df['return_5'].where(low_vol_mask, 0)
    volatility_regime_momentum = high_vol_momentum - low_vol_momentum
    
    # Intraday Pressure Persistence
    df['daily_pressure'] = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, np.nan)
    
    pressure_sign = np.sign(df['daily_pressure'])
    persistence = pressure_sign.groupby((pressure_sign != pressure_sign.shift(1)).cumsum()).cumcount() + 1
    intraday_pressure_persistence = df['daily_pressure'] * persistence
    
    # Volume-Weighted Range Efficiency
    df['range_efficiency'] = abs(df['close'] - df['open']) / df['true_range'].replace(0, np.nan)
    df['median_volume_20'] = df['volume'].rolling(window=20).median()
    df['volume_intensity'] = df['volume'] / df['median_volume_20'].replace(0, np.nan)
    volume_weighted_range_efficiency = df['range_efficiency'] * df['volume_intensity']
    
    # Multi-Timeframe Breakout Confirmation
    df['high_5'] = df['high'].rolling(window=5).max()
    df['high_10'] = df['high'].rolling(window=10).max()
    df['avg_volume_10'] = df['volume'].rolling(window=10).mean()
    
    short_breakout = (df['close'] > df['high_5']).astype(int)
    medium_breakout = (df['close'] > df['high_10']).astype(int)
    volume_confirmation = (df['volume'] > 1.8 * df['avg_volume_10']).astype(int)
    
    multi_timeframe_breakout = (short_breakout + medium_breakout) * volume_confirmation
    
    # Price-Volume Divergence Acceleration
    df['volume_change_3'] = df['volume'].pct_change(3)
    df['volume_change_6'] = df['volume'].pct_change(6)
    
    price_acceleration = df['return_3'] - df['return_6']
    volume_acceleration = df['volume_change_3'] - df['volume_change_6']
    
    price_volume_divergence = price_acceleration / (1 + abs(volume_acceleration))
    
    # Extreme Reversal Detection
    df['low_10'] = df['low'].rolling(window=10).min()
    df['high_10_roll'] = df['high'].rolling(window=10).max()
    df['volume_quantile_20'] = df['volume'].rolling(window=20).quantile(0.8)
    
    oversold = ((df['close'] == df['low_10']) & (df['volume'] > df['volume_quantile_20'])).astype(int)
    overbought = ((df['close'] == df['high_10_roll']) & (df['volume'] > df['volume_quantile_20'])).astype(int)
    
    extreme_reversal = oversold - overbought
    
    # Gap Filling Probability
    df['gap_size'] = abs(df['open'] - df['prev_close'])
    
    # Calculate gap filling days
    gap_filled = pd.Series(0, index=df.index)
    for i in range(1, len(df)):
        if df['gap_size'].iloc[i] > 0:
            current_gap = df['gap_size'].iloc[i]
            gap_direction = 1 if df['open'].iloc[i] > df['prev_close'].iloc[i] else -1
            
            for j in range(i+1, min(i+21, len(df))):
                if gap_direction > 0 and df['low'].iloc[j] <= df['prev_close'].iloc[i]:
                    gap_filled.iloc[i] = j - i
                    break
                elif gap_direction < 0 and df['high'].iloc[j] >= df['prev_close'].iloc[i]:
                    gap_filled.iloc[i] = j - i
                    break
    
    df['filling_speed'] = gap_filled / df['gap_size'].replace(0, np.nan)
    gap_filling_probability = -df['gap_size'] * df['filling_speed']
    
    # Volume Clustering Analysis
    df['avg_volume_20'] = df['volume'].rolling(window=20).mean()
    volume_cluster_mask = df['volume'] > 1.2 * df['avg_volume_20']
    
    cluster_groups = (volume_cluster_mask != volume_cluster_mask.shift(1)).cumsum()
    cluster_returns = df['close'].pct_change().groupby(cluster_groups).cumsum()
    cluster_counts = volume_cluster_mask.groupby(cluster_groups).transform('count')
    
    volume_clustering = cluster_returns / cluster_counts.replace(0, np.nan)
    
    # Regime-Sensitive Correlation
    df['trend_20'] = df['close'].rolling(window=20).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
    
    bull_mask = df['trend_20'] == 1
    bear_mask = df['trend_20'] == -1
    
    # Calculate rolling correlations
    bull_corr = pd.Series(index=df.index, dtype=float)
    bear_corr = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        if bull_mask.iloc[i]:
            window_data = df.iloc[i-19:i+1]
            bull_corr.iloc[i] = window_data['close'].corr(window_data['volume'])
        if bear_mask.iloc[i]:
            window_data = df.iloc[i-19:i+1]
            bear_corr.iloc[i] = window_data['close'].corr(window_data['volume'])
    
    regime_sensitive_correlation = bull_corr.fillna(0) - bear_corr.fillna(0)
    
    # Efficiency-Momentum Composite
    df['momentum_efficiency'] = df['return_5'] / df['atr_5'].replace(0, np.nan)
    efficiency_momentum_composite = df['range_efficiency'] * df['momentum_efficiency']
    
    # Combine all factors with equal weights
    factors = [
        volatility_regime_momentum,
        intraday_pressure_persistence,
        volume_weighted_range_efficiency,
        multi_timeframe_breakout,
        price_volume_divergence,
        extreme_reversal,
        gap_filling_probability,
        volume_clustering,
        regime_sensitive_correlation,
        efficiency_momentum_composite
    ]
    
    # Normalize and combine
    normalized_factors = []
    for factor in factors:
        normalized = (factor - factor.rolling(window=20).mean()) / factor.rolling(window=20).std().replace(0, 1)
        normalized_factors.append(normalized.fillna(0))
    
    # Equal-weighted combination
    result = sum(normalized_factors) / len(normalized_factors)
    
    return result
