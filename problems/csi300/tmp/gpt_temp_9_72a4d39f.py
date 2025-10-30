import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Timeframe Intraday Momentum Divergence with Volume-Efficiency Alignment
    """
    data = df.copy()
    
    # Multi-Timeframe Intraday Efficiency Divergence
    # Short-term intraday efficiency (5-day)
    data['intraday_range_5d'] = (data['high'] - data['low']).abs().rolling(window=5, min_periods=3).sum()
    data['net_return_5d'] = (data['close'] - data['open']).rolling(window=5, min_periods=3).sum()
    data['efficiency_5d'] = data['net_return_5d'] / (data['intraday_range_5d'] + 1e-8)
    
    # Medium-term intraday efficiency (10-day)
    data['intraday_range_10d'] = (data['high'] - data['low']).abs().rolling(window=10, min_periods=5).sum()
    data['net_return_10d'] = (data['close'] - data['open']).rolling(window=10, min_periods=5).sum()
    data['efficiency_10d'] = data['net_return_10d'] / (data['intraday_range_10d'] + 1e-8)
    
    # Long-term intraday efficiency (20-day)
    data['intraday_range_20d'] = (data['high'] - data['low']).abs().rolling(window=20, min_periods=10).sum()
    data['net_return_20d'] = (data['close'] - data['open']).rolling(window=20, min_periods=10).sum()
    data['efficiency_20d'] = data['net_return_20d'] / (data['intraday_range_20d'] + 1e-8)
    
    # Efficiency momentum divergence
    data['short_term_accel'] = data['efficiency_5d'] - data['efficiency_10d']
    data['medium_term_accel'] = data['efficiency_10d'] - data['efficiency_20d']
    data['efficiency_divergence'] = data['short_term_accel'] - data['medium_term_accel']
    
    # Volume-Efficiency Confirmation System
    # Volume momentum assessment
    data['volume_surge'] = data['volume'] / (data['volume'].shift(1) + 1e-8)
    
    # Volume persistence (count consecutive volume increases)
    volume_increase = data['volume'] > data['volume'].shift(1)
    data['volume_persistence'] = volume_increase.astype(int)
    for i in range(1, len(data)):
        if volume_increase.iloc[i]:
            data.iloc[i, data.columns.get_loc('volume_persistence')] = data['volume_persistence'].iloc[i-1] + 1
        else:
            data.iloc[i, data.columns.get_loc('volume_persistence')] = 0
    
    data['volume_breakout'] = data['volume_surge'] * data['volume_persistence']
    
    # Volume-weighted efficiency momentum
    data['volume_weighted_efficiency'] = data['efficiency_divergence'] * data['volume_breakout']
    
    # Gap Reversal with Volume-Efficiency Alignment
    data['gap_size'] = data['open'] - data['close'].shift(1)
    data['intraday_return'] = data['close'] - data['open']
    
    # Calculate reversal efficiency
    reversal_condition = ((data['gap_size'] > 0) & (data['intraday_return'] < 0)) | ((data['gap_size'] < 0) & (data['intraday_return'] > 0))
    data['reversal_efficiency'] = np.where(
        reversal_condition,
        data['intraday_return'] / (data['gap_size'].abs() + 1e-8),
        0
    )
    
    # Gap reversal confirmation
    data['gap_reversal_confirmation'] = data['reversal_efficiency'] * data['volume_breakout'] * np.sign(data['efficiency_divergence'])
    
    # Volatility-Regime Adaptive Scaling
    data['current_range'] = data['high'] - data['low']
    data['prev_range'] = data['current_range'].shift(1)
    data['range_expansion'] = data['current_range'] / (data['prev_range'] + 1e-8)
    
    # Volatility persistence (count consecutive range expansions)
    range_expand = data['range_expansion'] > 1
    data['volatility_persistence'] = range_expand.astype(int)
    for i in range(1, len(data)):
        if range_expand.iloc[i]:
            data.iloc[i, data.columns.get_loc('volatility_persistence')] = data['volatility_persistence'].iloc[i-1] + 1
        else:
            data.iloc[i, data.columns.get_loc('volatility_persistence')] = 0
    
    # Volatility scaling factor
    data['volatility_scaling'] = 1 / (data['range_expansion'] + 1e-8)
    data['volatility_adjusted_momentum'] = data['volume_weighted_efficiency'] * data['volatility_scaling']
    
    # Support/Resistance Break with Volume-Efficiency
    data['recent_support'] = data['low'].rolling(window=20, min_periods=10).min().shift(1)
    data['recent_resistance'] = data['high'].rolling(window=20, min_periods=10).max().shift(1)
    
    # Breakout detection
    resistance_break = data['close'] > data['recent_resistance']
    support_break = data['close'] < data['recent_support']
    
    # Breakout magnitude and confidence
    breakout_magnitude = np.where(
        resistance_break,
        (data['close'] - data['recent_resistance']) / (data['recent_resistance'] + 1e-8),
        np.where(
            support_break,
            (data['recent_support'] - data['close']) / (data['recent_support'] + 1e-8),
            0
        )
    )
    
    data['breakout_confidence'] = breakout_magnitude * data['volume_breakout'] * np.sign(data['efficiency_divergence'])
    
    # Composite Alpha Construction
    # Component integration with weights
    momentum_component = data['volume_weighted_efficiency'] * 0.4
    reversal_component = data['gap_reversal_confirmation'] * 0.25
    volatility_component = data['volatility_adjusted_momentum'] * 0.2
    breakout_component = data['breakout_confidence'] * 0.15
    
    # Quality assessment filters
    valid_efficiency = (data['intraday_range_5d'] > 0) & (data['intraday_range_10d'] > 0)
    valid_volume = data['volume'] > data['volume'].rolling(window=20).quantile(0.2)
    stable_regime = data['volatility_persistence'] < 5  # Avoid regime transitions
    
    # Final composite factor
    composite_factor = (
        momentum_component + reversal_component + volatility_component + breakout_component
    )
    
    # Apply quality filters
    composite_factor = np.where(
        valid_efficiency & valid_volume & stable_regime,
        composite_factor,
        composite_factor * 0.5  # Reduce signal strength for lower quality periods
    )
    
    # Dynamic scaling based on recent volatility
    recent_vol = data['current_range'].rolling(window=10).std()
    scaling_factor = 1 / (recent_vol + 1e-8)
    composite_factor = composite_factor * scaling_factor
    
    # Normalize and return
    factor_series = pd.Series(composite_factor, index=data.index)
    factor_series = (factor_series - factor_series.rolling(window=20).mean()) / (factor_series.rolling(window=20).std() + 1e-8)
    
    return factor_series
