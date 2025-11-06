import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Generate alpha factor combining regime-adaptive momentum divergence, efficiency-volume reversal detection,
    flow persistence with regime context, and breakout quality with multi-timeframe confirmation.
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Regime-Adaptive Momentum Divergence
    # Price Momentum Components
    data['return_5d'] = data['close'] / data['close'].shift(5) - 1
    data['return_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_acceleration'] = data['return_5d'] - data['return_10d']
    
    # Volume Divergence
    data['volume_change_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_direction'] = np.sign(data['volume_change_5d']) * np.sign(data['return_5d'])
    data['divergence_strength'] = data['return_5d'] - data['volume_change_5d']
    
    # Volatility Regime Detection
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['volatility_20d'] = data['true_range'].rolling(window=20).mean()
    data['volatility_ratio'] = data['true_range'] / data['volatility_20d']
    
    # Combined Factor 1
    data['base_signal'] = data['momentum_acceleration'] * data['divergence_strength']
    data['adjusted_signal'] = data['base_signal'] / data['volatility_ratio']
    data['factor1'] = data['adjusted_signal'] * data['volume_direction']
    
    # 2. Efficiency-Volume Reversal Detection
    # Price Efficiency Analysis
    data['daily_range_efficiency'] = abs(data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])
    data['efficiency_5d_avg'] = data['daily_range_efficiency'].rolling(window=5).mean()
    data['efficiency_trend'] = data['daily_range_efficiency'] / data['efficiency_5d_avg']
    
    # Volume Anomaly Detection
    data['volume_20d_median'] = data['volume'].rolling(window=20).median()
    data['volume_spike_magnitude'] = data['volume'] / data['volume'].shift(10)
    
    # Factor 2
    data['factor2'] = (1 - data['efficiency_trend']) * data['volume_spike_magnitude']
    
    # 3. Flow Persistence with Regime Context
    # Order Flow Calculation
    data['flow_direction'] = np.where(data['close'] > data['close'].shift(1), 1, -1)
    data['flow_magnitude'] = data['amount'] / data['amount'].rolling(window=20).mean()
    
    # Flow persistence calculation
    data['flow_persistence'] = 0
    for i in range(1, len(data)):
        if data['flow_direction'].iloc[i] == data['flow_direction'].iloc[i-1]:
            data['flow_persistence'].iloc[i] = data['flow_persistence'].iloc[i-1] + 1
        else:
            data['flow_persistence'].iloc[i] = 1
    
    # Factor 3
    data['factor3'] = data['flow_persistence'] * data['flow_magnitude'] / data['volatility_ratio']
    
    # 4. Breakout Quality with Multi-Timeframe Confirmation
    # Breakout Identification
    data['max_20d'] = data['close'].rolling(window=20).max().shift(1)
    data['price_breakout'] = data['close'] > data['max_20d']
    
    data['range_20d_avg'] = (data['high'] - data['low']).rolling(window=20).mean()
    data['range_expansion'] = (data['high'] - data['low']) > (1.2 * data['range_20d_avg'])
    
    data['gap_confirmation'] = data['open'] > (data['close'].shift(1) * 1.01)
    
    # Volume-Volatility Alignment
    data['volume_20d_avg'] = data['volume'].rolling(window=20).mean()
    data['volume_ratio'] = data['volume'] / data['volume_20d_avg']
    
    data['pre_breakout_volatility'] = data['volatility_20d'].shift(1)
    
    # Multi-timeframe momentum alignment
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_alignment'] = np.sign(data['momentum_5d']) == np.sign(data['momentum_10d'])
    
    # Breakout strength calculation
    data['breakout_strength'] = (data['close'] - data['max_20d']) / data['max_20d']
    
    # Factor 4
    data['factor4'] = data['breakout_strength'] * data['volume_ratio'] / data['pre_breakout_volatility']
    data['factor4'] = data['factor4'] * data['momentum_alignment']
    
    # Combine all factors with equal weighting
    factors = ['factor1', 'factor2', 'factor3', 'factor4']
    for factor in factors:
        data[factor] = (data[factor] - data[factor].mean()) / data[factor].std()
    
    data['combined_factor'] = (
        data['factor1'] * 0.25 + 
        data['factor2'] * 0.25 + 
        data['factor3'] * 0.25 + 
        data['factor4'] * 0.25
    )
    
    return data['combined_factor']
