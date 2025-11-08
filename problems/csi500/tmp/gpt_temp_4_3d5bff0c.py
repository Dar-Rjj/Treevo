import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Momentum-Range Consistency
    # Short-term momentum persistence (2-day directional streak)
    close_returns = data['close'].pct_change()
    momentum_direction = np.sign(close_returns)
    momentum_streak = momentum_direction.rolling(window=2).apply(lambda x: 1 if len(set(x)) == 1 and x.iloc[0] != 0 else 0, raw=False)
    
    # Medium-term momentum deceleration (5-day vs 12-day momentum difference)
    mom_5day = data['close'].pct_change(5)
    mom_12day = data['close'].pct_change(12)
    momentum_deceleration = mom_5day - mom_12day
    
    # Momentum consistency score (persistence × deceleration)
    momentum_consistency = momentum_streak * momentum_deceleration
    
    # Range-Volume Dynamics
    # Range compression ratio (2-day vs 10-day average range)
    daily_range = (data['high'] - data['low']) / data['close']
    range_2day = daily_range.rolling(window=2).mean()
    range_10day = daily_range.rolling(window=10).mean()
    range_compression = range_2day / range_10day
    
    # Volume-range concentration (absolute return / range × volume)
    abs_return = np.abs(data['close'].pct_change())
    volume_range_concentration = abs_return / (daily_range * data['volume'])
    
    # Range-volume convergence (compression × concentration)
    range_volume_convergence = range_compression * volume_range_concentration
    
    # Volume-Weighted Reversal
    # Daily reversal magnitude (close-open relative to range)
    reversal_magnitude = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Volume-adjusted reversal persistence (2-day consistency × magnitude)
    reversal_direction = np.sign(reversal_magnitude)
    reversal_streak = reversal_direction.rolling(window=2).apply(lambda x: 1 if len(set(x)) == 1 and x.iloc[0] != 0 else 0, raw=False)
    volume_adjusted_reversal = reversal_streak * reversal_magnitude * data['volume']
    
    # Reversal-range alignment (direction vs range compression)
    reversal_range_alignment = np.sign(reversal_magnitude) * range_compression
    
    # Volume-Regime Adaptive Combination
    # Volume regime detection (8-day average volume classification)
    volume_8day_avg = data['volume'].rolling(window=8).mean()
    volume_regime = pd.cut(volume_8day_avg, bins=3, labels=[1, 2, 3])
    volume_regime = volume_regime.fillna(2).astype(int)
    
    # Regime-adaptive component weighting
    regime_weights = {
        1: {'momentum': 0.4, 'range': 0.3, 'reversal': 0.3},  # Low volume
        2: {'momentum': 0.3, 'range': 0.4, 'reversal': 0.3},  # Medium volume
        3: {'momentum': 0.2, 'range': 0.3, 'reversal': 0.5}   # High volume
    }
    
    # Final composite factor (momentum × range × reversal × regime weights)
    factor_values = []
    for idx, date in enumerate(data.index):
        regime = volume_regime.loc[date]
        weights = regime_weights[regime]
        
        momentum_component = momentum_consistency.loc[date] if not pd.isna(momentum_consistency.loc[date]) else 0
        range_component = range_volume_convergence.loc[date] if not pd.isna(range_volume_convergence.loc[date]) else 0
        reversal_component = (volume_adjusted_reversal.loc[date] + reversal_range_alignment.loc[date]) / 2 if not pd.isna(volume_adjusted_reversal.loc[date]) else 0
        
        composite_factor = (
            weights['momentum'] * momentum_component +
            weights['range'] * range_component +
            weights['reversal'] * reversal_component
        )
        factor_values.append(composite_factor)
    
    factor_series = pd.Series(factor_values, index=data.index)
    return factor_series
