import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Efficiency Gap Momentum Synthesis alpha factor
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
    
    # Gap Absorption Efficiency Framework
    data['gap_size'] = abs(data['open'] - data['prev_close']) / (data['high'].shift(1) - data['low'].shift(1))
    data['gap_absorption'] = (data['close'] - data['open']) / abs(data['open'] - data['prev_close'])
    data['gap_fill_pct'] = np.where(
        data['open'] > data['prev_close'],
        (data['close'] - data['prev_close']) / (data['open'] - data['prev_close']),
        (data['prev_close'] - data['close']) / (data['prev_close'] - data['open'])
    )
    
    # Gap Volume Participation
    data['morning_vol_concentration'] = data['volume'] / (
        data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)
    )
    data['vol_vs_5day_avg'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['vol_gap_direction'] = np.sign(data['close'] - data['open']) * data['volume']
    
    # Volatility-Regime Integration
    data['atr_5'] = data['true_range'].rolling(window=5).mean()
    data['atr_20'] = data['true_range'].rolling(window=20).mean()
    data['range_vs_10day'] = (data['high'] - data['low']) / (data['high'] - data['low']).rolling(window=10).mean()
    data['vol_compression'] = data['atr_5'] / data['atr_20']
    
    # Range breakout detection
    data['high_20'] = data['high'].rolling(window=20).max()
    data['low_20'] = data['low'].rolling(window=20).min()
    data['range_breakout'] = np.where(
        data['close'] > data['high_20'], 1,
        np.where(data['close'] < data['low_20'], -1, 0)
    )
    data['vol_gap_interaction'] = data['gap_size'] * data['vol_compression']
    
    # Momentum-Pressure Convergence
    data['intraday_momentum'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['multi_day_momentum'] = (data['close'] - data['close'].shift(3)) / data['true_range']
    data['momentum_decay'] = (data['close'] / data['close'].shift(3) - 1) - (data['close'] / data['close'].shift(5) - 1)
    
    # Pressure Accumulation System
    data['daily_pressure'] = ((data['close'] - data['low']) - (data['high'] - data['close'])) / (data['high'] - data['low'])
    data['cumulative_pressure'] = (
        data['volume'] * (data['close'] - data['open']) +
        data['volume'].shift(1) * (data['close'].shift(1) - data['open'].shift(1)) +
        data['volume'].shift(2) * (data['close'].shift(2) - data['open'].shift(2)) +
        data['volume'].shift(3) * (data['close'].shift(3) - data['open'].shift(3)) +
        data['volume'].shift(4) * (data['close'].shift(4) - data['open'].shift(4))
    )
    data['vol_acceleration'] = data['volume'] / data['volume'].rolling(window=3).mean()
    
    # Pressure persistence
    data['pressure_sign'] = np.sign(data['daily_pressure'])
    data['pressure_persistence'] = 0
    for i in range(1, len(data)):
        if data['pressure_sign'].iloc[i] == data['pressure_sign'].iloc[i-1]:
            data['pressure_persistence'].iloc[i] = data['pressure_persistence'].iloc[i-1] + 1
        else:
            data['pressure_persistence'].iloc[i] = 1
    
    # Efficiency-Volume Alignment
    data['price_efficiency'] = abs(data['close'] - data['open']) / data['true_range']
    data['efficiency_trend'] = data['price_efficiency'] / data['price_efficiency'].shift(3)
    data['efficiency_change'] = data['price_efficiency'] - data['price_efficiency'].shift(3)
    
    data['vol_trend_alignment'] = data['volume'] / data['volume'].rolling(window=5).mean()
    data['vol_momentum_clustering'] = data['morning_vol_concentration'] * abs(data['intraday_momentum'])
    data['vol_pressure_interaction'] = data['daily_pressure'] * data['vol_acceleration']
    data['vol_gap_divergence'] = data['vol_acceleration'] - data['gap_absorption']
    
    # Multi-Timeframe Gap Validation
    data['short_term_gap'] = data['close'] - data['prev_close']
    data['medium_term_gap'] = data['close'] - data['close'].shift(3)
    data['gap_sustainability'] = data['short_term_gap'] / data['medium_term_gap']
    
    data['gap_size_norm'] = abs(data['open'] - data['prev_close']) / data['atr_20']
    data['relative_amplitude'] = (data['high'] - data['low']) / data['close']
    data['amplitude_gap_interaction'] = data['gap_size'] / data['relative_amplitude']
    data['multi_timeframe_efficiency'] = data['gap_absorption'] * data['gap_size_norm']
    
    # Exhaustion Pattern Detection
    data['vol_spike_decay'] = np.where(
        (data['volume'] / data['volume'].shift(1) > 2.0) & 
        (np.sign(data['daily_pressure']) != np.sign(data['daily_pressure'].shift(1))), 1, 0
    )
    data['price_rejection'] = (data['high'] - data['close']) / data['true_range']
    data['overextension'] = abs(data['close'] - data['close'].shift(5)) / data['atr_5']
    
    # Primary Signal Components
    data['absorption_vol_comp'] = data['gap_absorption'] * data['vol_compression']
    data['momentum_pressure'] = data['intraday_momentum'] * data['cumulative_pressure']
    data['volume_efficiency'] = data['vol_trend_alignment'] * data['efficiency_change']
    
    # Secondary Signal Components
    data['pressure_vol_gap'] = data['daily_pressure'] * data['vol_acceleration'] * data['gap_absorption']
    
    # Final Alpha Factor Synthesis
    # Core integration with regime-adaptive scaling
    data['vol_efficiency_gap_momentum'] = (
        data['absorption_vol_comp'] * 0.3 +
        data['momentum_pressure'] * 0.25 +
        data['volume_efficiency'] * 0.2 +
        data['multi_timeframe_efficiency'] * 0.15 +
        data['pressure_vol_gap'] * 0.1
    )
    
    # Apply exhaustion filters
    exhaustion_filter = np.where(
        (data['vol_spike_decay'] == 1) |
        (data['price_rejection'] > 0.7) |
        (data['overextension'] > 2.0),
        0.5, 1.0
    )
    
    # Apply range breakout enhancement
    breakout_enhancement = np.where(data['range_breakout'] != 0, 1.2, 1.0)
    
    # Final factor with adjustments
    final_factor = data['vol_efficiency_gap_momentum'] * exhaustion_filter * breakout_enhancement
    
    return final_factor
