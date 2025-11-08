import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Pressure Accumulation Factor
    # Calculate pressure accumulation components
    data['buying_pressure'] = np.maximum(0, data['close'] - data['open']) * data['volume']
    data['selling_pressure'] = np.maximum(0, data['open'] - data['close']) * data['volume']
    
    data['buying_pressure_3d'] = data['buying_pressure'].rolling(window=3, min_periods=1).sum()
    data['selling_pressure_3d'] = data['selling_pressure'].rolling(window=3, min_periods=1).sum()
    
    # Analyze pressure imbalance patterns
    total_pressure = data['buying_pressure_3d'] + data['selling_pressure_3d']
    data['net_pressure_ratio'] = np.where(total_pressure > 0, 
                                         (data['buying_pressure_3d'] - data['selling_pressure_3d']) / total_pressure, 
                                         0)
    
    # Pressure persistence
    data['pressure_direction'] = np.where(data['buying_pressure'] > data['selling_pressure'], 1, 
                                         np.where(data['selling_pressure'] > data['buying_pressure'], -1, 0))
    
    pressure_persistence = []
    current_streak = 0
    current_direction = 0
    
    for direction in data['pressure_direction']:
        if direction == current_direction and direction != 0:
            current_streak += 1
        else:
            current_streak = 1 if direction != 0 else 0
            current_direction = direction
        pressure_persistence.append(current_streak)
    
    data['pressure_persistence'] = pressure_persistence
    
    # Assess intraday pressure confirmation
    data['prev_close'] = data['close'].shift(1)
    data['opening_pressure'] = np.sign(data['open'] - data['prev_close']) * data['volume']
    data['closing_pressure'] = np.sign(data['close'] - data['open']) * data['volume']
    
    # Generate pressure accumulation signal
    persistence_strength = np.minimum(data['pressure_persistence'] / 5, 1.0)
    data['pressure_signal'] = data['net_pressure_ratio'] * persistence_strength * (data['opening_pressure'] + data['closing_pressure'])
    
    # Apply volume context adjustment
    data['volume_10d_avg'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['volume_ratio'] = data['volume'] / data['volume_10d_avg']
    
    # Volume trend (3-day slope)
    data['volume_trend'] = data['volume'].rolling(window=3).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0, 
        raw=False
    )
    
    # Final factor construction
    volume_context = data['volume_ratio'] * (1 + data['volume_trend'])
    directional_momentum = np.sign(data['close'] - data['close'].shift(3)).fillna(0)
    
    intraday_pressure_factor = data['pressure_signal'] * volume_context * directional_momentum
    
    # Range Breakout Efficiency Factor
    # Calculate breakout efficiency components
    data['range'] = data['high'] - data['low']
    data['range_5d_avg'] = data['range'].rolling(window=5, min_periods=1).mean()
    data['range_expansion'] = data['range'] / data['range_5d_avg']
    
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['breakout_magnitude'] = np.maximum(
        data['high'] - data['prev_high'], 
        data['prev_low'] - data['low']
    ).fillna(0)
    
    # Analyze breakout quality patterns
    data['clean_breakout'] = np.where(data['range'] > 0, 
                                     data['breakout_magnitude'] / data['range'], 
                                     0)
    data['follow_through'] = np.where(data['breakout_magnitude'] > 0, 
                                     (data['close'] - data['open']) / data['breakout_magnitude'], 
                                     0)
    
    # Assess volume confirmation
    data['breakout_volume_ratio'] = data['volume'] / data['volume_10d_avg']
    
    # Generate efficiency score
    breakout_quality = data['clean_breakout'] * data['follow_through']
    data['efficiency_score'] = data['range_expansion'] * breakout_quality * data['breakout_volume_ratio']
    
    # Apply trend context
    data['short_term_trend'] = np.sign(data['close'] - data['close'].shift(3)).fillna(0)
    data['medium_term_trend'] = np.sign(data['close'] - data['close'].shift(10)).fillna(0)
    
    # Final factor output
    trend_alignment = np.where(data['short_term_trend'] == data['medium_term_trend'], 1.5, 1.0)
    volatility_persistence = data['range'].rolling(window=5).std() / data['range_5d_avg']
    volatility_filter = np.where(volatility_persistence > volatility_persistence.rolling(window=10).mean(), 1.0, 0.5)
    
    breakout_efficiency_factor = data['efficiency_score'] * trend_alignment * volatility_filter
    
    # Volatility Compression Expansion Factor
    # Identify compression conditions
    data['range_10d_avg'] = data['range'].rolling(window=10, min_periods=1).mean()
    data['range_compression'] = data['range'] / data['range_10d_avg']
    data['volume_compression'] = data['volume'] / data['volume_10d_avg']
    
    # Analyze expansion signals
    data['directional_expansion'] = np.where(data['range'] > 0, 
                                            np.abs(data['close'] - data['open']) / data['range'], 
                                            0)
    
    data['volume_3d_avg'] = data['volume'].rolling(window=3, min_periods=1).mean()
    data['volume_acceleration'] = (data['volume_3d_avg'] - data['volume_10d_avg']) / data['volume_10d_avg']
    
    # Assess compression-expansion timing
    expansion_threshold = data['range_compression'].rolling(window=20).quantile(0.7)
    data['is_expansion'] = (data['range_compression'] > expansion_threshold).astype(int)
    
    compression_duration = []
    current_duration = 0
    
    for is_exp in data['is_expansion']:
        if is_exp == 0:
            current_duration += 1
        else:
            current_duration = 0
        compression_duration.append(current_duration)
    
    data['compression_duration'] = compression_duration
    
    # Generate expansion anticipation score
    compression_intensity = 1.0 / (1.0 + data['range_compression'])
    expansion_signals = data['directional_expansion'] * (1 + data['volume_acceleration'])
    timing_factor = np.minimum(data['compression_duration'] / 10, 1.0)
    
    data['expansion_anticipation'] = compression_intensity * expansion_signals * timing_factor
    
    # Apply price level context
    data['relative_position'] = np.where(data['range'] > 0, 
                                        (data['close'] - data['low']) / data['range'], 
                                        0.5)
    
    # Support/resistance proximity (simplified as distance from recent extremes)
    data['resistance_proximity'] = (data['high'].rolling(window=10).max() - data['close']) / data['range_10d_avg']
    data['support_proximity'] = (data['close'] - data['low'].rolling(window=10).min()) / data['range_10d_avg']
    
    price_context = 1.0 + 0.5 * (data['relative_position'] - 0.5) - 0.2 * np.minimum(data['resistance_proximity'], data['support_proximity'])
    
    # Final factor construction
    directional_bias = np.sign(data['close'] - data['close'].shift(5)).fillna(0)
    volatility_compression_factor = data['expansion_anticipation'] * price_context * (1 + 0.1 * directional_bias)
    
    # Momentum Fragmentation Factor
    # Calculate momentum fragmentation components
    data['intraday_up_move'] = data['high'] - data['close']
    data['intraday_down_move'] = data['close'] - data['low']
    data['intraday_asymmetry'] = np.where(
        (data['intraday_up_move'] + data['intraday_down_move']) > 0,
        np.abs(data['intraday_up_move'] - data['intraday_down_move']) / (data['intraday_up_move'] + data['intraday_down_move']),
        0
    )
    
    data['opening_gap'] = (data['open'] - data['prev_close']).abs() / data['prev_close']
    data['closing_momentum'] = (data['close'] - data['open']).abs() / data['open']
    data['interday_fragmentation'] = np.abs(data['opening_gap'] - data['closing_momentum'])
    
    # Analyze fragmentation patterns
    data['momentum_consistency'] = data['intraday_asymmetry'].rolling(window=5).corr(data['interday_fragmentation'])
    
    # Fragmentation persistence (days with similar high fragmentation)
    fragmentation_threshold = data['intraday_asymmetry'].rolling(window=20).quantile(0.7)
    high_fragmentation = (data['intraday_asymmetry'] > fragmentation_threshold).astype(int)
    
    fragmentation_persistence = []
    current_streak = 0
    
    for is_high in high_fragmentation:
        if is_high == 1:
            current_streak += 1
        else:
            current_streak = 0
        fragmentation_persistence.append(current_streak)
    
    data['fragmentation_persistence'] = fragmentation_persistence
    
    # Generate fragmentation score
    price_fragmentation = data['intraday_asymmetry'] * data['interday_fragmentation']
    persistence_weight = np.minimum(data['fragmentation_persistence'] / 5, 1.0)
    consistency_weight = 1.0 - np.abs(data['momentum_consistency'])
    
    data['fragmentation_score'] = price_fragmentation * persistence_weight * consistency_weight
    
    # Apply market regime context
    volatility_regime = data['range_10d_avg'] / data['range_10d_avg'].rolling(window=20).mean()
    trend_regime = data['close'].rolling(window=10).apply(
        lambda x: (x.iloc[-1] - x.iloc[0]) / x.iloc[0] if x.iloc[0] != 0 else 0,
        raw=False
    ).abs()
    
    regime_normalization = 1.0 / (1.0 + 0.5 * volatility_regime + 0.3 * trend_regime)
    
    # Final factor output with reversal probability mapping
    reversal_probability = 1.0 / (1.0 + np.exp(-2 * (data['fragmentation_score'] - 0.5)))
    momentum_fragmentation_factor = data['fragmentation_score'] * regime_normalization * reversal_probability
    
    # Combine all factors with equal weighting
    combined_factor = (
        intraday_pressure_factor.fillna(0) + 
        breakout_efficiency_factor.fillna(0) + 
        volatility_compression_factor.fillna(0) + 
        momentum_fragmentation_factor.fillna(0)
    ) / 4
    
    return combined_factor
