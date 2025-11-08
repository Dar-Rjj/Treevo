import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Intraday Momentum-Volume Divergence Factor
    Combines momentum divergence, volume patterns, price range utilization, and market regime adaptation
    """
    data = df.copy()
    
    # Calculate Momentum Divergence Components
    # Short vs Medium-term Momentum Gap
    data['momentum_short'] = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    data['momentum_medium'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    
    # Momentum Acceleration Profile
    data['momentum_change_rate'] = (data['momentum_short'] - data['momentum_medium']) / (np.abs(data['momentum_medium']) + 1e-8)
    
    # Momentum Persistence
    momentum_sign = np.sign(data['momentum_short'])
    momentum_sign_shift = momentum_sign.shift(1)
    data['momentum_persistence'] = (momentum_sign == momentum_sign_shift).astype(int)
    data['momentum_persistence'] = data['momentum_persistence'].rolling(window=5, min_periods=1).sum()
    
    # Momentum Volatility Ratio
    momentum_vol = data['momentum_short'].rolling(window=5, min_periods=3).std()
    data['momentum_vol_ratio'] = data['momentum_short'] / (momentum_vol + 1e-8)
    
    # Calculate Volume Divergence Patterns
    # Volume-Momentum Decoupling
    data['volume_adjusted_momentum'] = data['momentum_short'] * data['volume']
    data['volume_trend'] = (data['volume'] - data['volume'].shift(5)) / (data['volume'].shift(5) + 1e-8)
    
    # Volume Distribution Skew
    high_low_range = data['high'] - data['low']
    data['volume_losing_territory'] = ((data['high'] - data['close']) / (high_low_range + 1e-8)) * data['volume']
    data['volume_winning_territory'] = ((data['close'] - data['low']) / (high_low_range + 1e-8)) * data['volume']
    data['volume_efficiency_asymmetry'] = data['volume_winning_territory'] / (data['volume_losing_territory'] + 1e-8)
    
    # Volume Clustering Indicator
    volume_ma = data['volume'].rolling(window=5, min_periods=3).mean()
    high_volume = (data['volume'] > volume_ma).astype(int)
    data['volume_spike_persistence'] = high_volume.rolling(window=3, min_periods=1).sum()
    
    # Volume Return Correlation
    volume_change = (data['volume'] - data['volume'].shift(1)) / (data['volume'].shift(1) + 1e-8)
    daily_return = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['volume_return_correlation'] = volume_change * daily_return
    
    # Calculate Price Range Utilization
    # Intraday Range Efficiency
    data['range_efficiency'] = np.abs(data['close'] - data['open']) / (high_low_range + 1e-8)
    
    # Range Expansion Patterns
    data['range'] = high_low_range
    data['range_change_momentum'] = (data['range'] - data['range'].shift(1)) / (data['range'].shift(1) + 1e-8)
    range_ma = data['range'].rolling(window=5, min_periods=3).mean()
    data['range_volatility_ratio'] = data['range'] / (range_ma + 1e-8)
    
    # Overnight Gap Impact
    overnight_gap = np.abs(data['open'] - data['close'].shift(1))
    intraday_move = np.abs(data['close'] - data['open'])
    data['gap_fill_efficiency'] = intraday_move / (overnight_gap + 1e-8)
    
    gap_direction = np.sign(data['open'] - data['close'].shift(1))
    intraday_direction = np.sign(data['close'] - data['open'])
    data['gap_direction_consistency'] = (gap_direction == intraday_direction).astype(float)
    
    # Integrate Divergence Signals
    # Combine Momentum-Volume Mismatch
    data['momentum_volume_divergence'] = data['momentum_change_rate'] * data['volume_trend']
    data['volume_confirmed_momentum'] = data['momentum_short'] * data['volume_efficiency_asymmetry']
    
    # Apply Range Context Filters
    data['divergence_range_weighted'] = data['momentum_volume_divergence'] * data['range_efficiency']
    data['divergence_gap_adjusted'] = data['divergence_range_weighted'] * data['gap_fill_efficiency']
    
    # Calculate Market Regime Adaptation
    # Volatility Regime Indicator
    range_20ma = data['range'].rolling(window=20, min_periods=10).mean()
    data['volatility_regime'] = data['range'] / (range_20ma + 1e-8)
    data['volatility_trend'] = data['range'].rolling(window=5, min_periods=3).apply(lambda x: (x[-1] - x[0]) / x[0] if x[0] != 0 else 0)
    
    # Trend Regime Classification
    data['trend_strength'] = np.abs(data['momentum_medium'])
    price_trend = np.sign(data['close'] - data['close'].shift(5))
    data['trend_consistency'] = (price_trend == price_trend.shift(1)).astype(int)
    data['trend_consistency'] = data['trend_consistency'].rolling(window=5, min_periods=1).sum()
    
    # Final Factor Construction
    # Generate Composite Divergence Score
    base_divergence = data['divergence_gap_adjusted'] * data['volume_confirmed_momentum']
    
    # Apply Market Regime Weights
    volatility_weight = 1.0 / (1.0 + np.abs(data['volatility_regime'] - 1.0))
    trend_weight = data['trend_consistency'] / 5.0
    
    # Calculate Normalized Divergence Measure
    composite_score = base_divergence * volatility_weight * trend_weight
    
    # Normalize using rolling z-score
    factor_mean = composite_score.rolling(window=20, min_periods=10).mean()
    factor_std = composite_score.rolling(window=20, min_periods=10).std()
    final_factor = (composite_score - factor_mean) / (factor_std + 1e-8)
    
    return final_factor
