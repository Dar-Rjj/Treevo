import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Efficiency Gap Momentum with Multi-Regime Volume Confirmation
    """
    data = df.copy()
    
    # Calculate basic price metrics
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['prev_close']),
            abs(data['low'] - data['prev_close'])
        )
    )
    
    # Gap Dynamics
    data['prev_high'] = data['high'].shift(1)
    data['prev_low'] = data['low'].shift(1)
    data['prev_range'] = data['prev_high'] - data['prev_low']
    data['gap_size'] = (data['open'] - data['prev_close']) / np.maximum(data['prev_range'], 0.001)
    data['absorption_efficiency'] = (data['close'] - data['open']) / np.maximum(abs(data['open'] - data['prev_close']), 0.001)
    data['gap_fill_completion'] = np.where(
        data['open'] > data['prev_close'],
        (data['high'] - data['open']) / np.maximum(data['high'] - data['prev_close'], 0.001),
        (data['open'] - data['low']) / np.maximum(data['prev_close'] - data['low'], 0.001)
    )
    
    # Volume-Gap Interaction
    data['volume_3day_sum'] = data['volume'].rolling(window=3, min_periods=1).sum().shift(1)
    data['morning_volume_concentration'] = data['volume'] / np.maximum(data['volume_3day_sum'], 1)
    data['volume_5day_avg'] = data['volume'].rolling(window=5, min_periods=1).mean().shift(1)
    data['gap_day_volume_ratio'] = data['volume'] / np.maximum(data['volume_5day_avg'], 1)
    
    # Multi-Timeframe Momentum Integration
    data['short_momentum'] = data['close'] / data['close'].shift(5) - 1
    data['medium_momentum'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_acceleration'] = data['short_momentum'] - data['medium_momentum']
    
    # Gap Persistence Validation
    data['short_gap_impact'] = (data['close'] - data['close'].shift(1)) / np.maximum(data['true_range'].shift(1), 0.001)
    data['medium_gap_validation'] = (data['close'] - data['close'].shift(3)) / np.maximum(data['true_range'].rolling(window=3, min_periods=1).mean().shift(1), 0.001)
    data['gap_sustainability'] = np.where(
        data['short_gap_impact'] * data['medium_gap_validation'] > 0,
        abs(data['short_gap_impact']) / np.maximum(abs(data['medium_gap_validation']), 0.001),
        0
    )
    
    # Volatility-Efficiency Dynamics
    data['atr_5day'] = data['true_range'].rolling(window=5, min_periods=1).mean()
    data['volatility_change_3day'] = data['atr_5day'] / data['atr_5day'].shift(3) - 1
    data['range_10day_avg'] = (data['high'] - data['low']).rolling(window=10, min_periods=1).mean()
    data['range_ratio'] = (data['high'] - data['low']) / np.maximum(data['range_10day_avg'], 0.001)
    
    # Efficiency Component
    data['price_efficiency_ratio'] = abs(data['close'] - data['open']) / np.maximum(data['true_range'], 0.001)
    data['efficiency_change_3day'] = data['price_efficiency_ratio'] / data['price_efficiency_ratio'].shift(3) - 1
    
    # Volatility-Efficiency Gap Divergence
    data['volatility_efficiency_gap'] = data['volatility_change_3day'] * data['absorption_efficiency']
    
    # Multi-Regime Volume Analysis
    # Volume Persistence Analysis
    data['volume_direction'] = np.where(
        data['volume'] > data['volume'].shift(1), 1,
        np.where(data['volume'] < data['volume'].shift(1), -1, 0)
    )
    
    # Calculate volume persistence streak
    volume_streak = []
    current_streak = 0
    current_direction = 0
    
    for i, direction in enumerate(data['volume_direction']):
        if direction == 0:
            current_streak = 0
            current_direction = 0
        elif direction == current_direction:
            current_streak += direction
        else:
            current_streak = direction
            current_direction = direction
        volume_streak.append(current_streak)
    
    data['volume_persistence'] = volume_streak
    
    data['volume_20day_avg'] = data['volume'].rolling(window=20, min_periods=1).mean()
    data['volume_breakout_strength'] = data['volume'] / np.maximum(data['volume_20day_avg'], 1)
    
    # Volume-Price Gap Alignment
    data['volume_momentum_5day'] = data['volume'] / np.maximum(data['volume'].shift(5), 1) - 1
    data['gap_price_momentum'] = (data['close'] - data['close'].shift(1)) / np.maximum(data['true_range'].shift(1), 0.001)
    data['volume_gap_divergence'] = data['volume_momentum_5day'] - data['gap_price_momentum']
    
    # Volume-Efficiency Gap Interaction
    data['volume_efficiency_interaction'] = data['volume_gap_divergence'] * data['price_efficiency_ratio']
    
    # Regime Identification and Breakout Detection
    data['daily_return'] = data['close'].pct_change()
    data['volatility_20day'] = data['daily_return'].rolling(window=20, min_periods=1).std()
    data['volatility_percentile'] = data['volatility_20day'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 80)) if len(x.dropna()) > 0 else 0
    )
    
    data['daily_amplitude'] = (data['high'] - data['low']) / data['close']
    data['amplitude_percentile'] = data['daily_amplitude'].rolling(window=20, min_periods=1).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 80)) if len(x.dropna()) > 0 else 0
    )
    
    # Gap Breakout Conditions
    data['volume_breakout_flag'] = (data['volume'] > 1.5 * data['volume_20day_avg']).astype(int)
    data['range_breakout_flag'] = ((data['high'] - data['low']) / data['prev_close'] > 1.5 * data['range_10day_avg'] / data['prev_close']).astype(int)
    data['combined_breakout'] = data['volume_breakout_flag'] * data['range_breakout_flag']
    
    # Composite Signal Generation
    # Core Gap Absorption Signal
    core_gap_signal = (
        data['absorption_efficiency'] * 
        data['morning_volume_concentration'] * 
        (1 + data['momentum_acceleration']) *
        np.sign(data['volatility_efficiency_gap'])
    )
    
    # Multi-Regime Volume Confirmation
    volume_confirmation = (
        np.sign(data['volume_persistence']) *
        data['volume_gap_divergence'] *
        data['volume_efficiency_interaction']
    )
    
    # Regime-Adaptive Enhancement
    regime_enhancement = (
        (1 + data['volatility_percentile'] * 0.5) *
        (1 + data['amplitude_percentile'] * 0.3)
    )
    
    # Breakout Signal Integration
    breakout_enhancement = (1 + data['combined_breakout'] * 0.8)
    
    # Final Alpha Factor
    alpha_factor = (
        core_gap_signal *
        (1 + volume_confirmation * 0.5) *
        regime_enhancement *
        breakout_enhancement
    )
    
    # Clean up and return
    alpha_series = pd.Series(alpha_factor, index=data.index)
    alpha_series = alpha_series.replace([np.inf, -np.inf], np.nan)
    
    return alpha_series
