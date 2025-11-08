import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum Analysis
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    data['abs_momentum_5d'] = data['momentum_5d'].abs()
    data['abs_momentum_10d'] = data['momentum_10d'].abs()
    
    # Momentum Acceleration System
    data['momentum_ratio'] = (data['close'] / data['close'].shift(5)) / (data['close'] / data['close'].shift(10))
    data['acceleration'] = data['momentum_ratio'] - 2 * data['momentum_ratio'].shift(1) + data['momentum_ratio'].shift(2)
    data['acceleration_direction'] = np.sign(data['acceleration'])
    
    # Volume Dynamics Integration
    data['volume_momentum'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_ratio'] = data['volume'].rolling(5).mean() / data['volume'].rolling(10).mean()
    data['volume_divergence'] = data['volume_momentum'] - data['volume_ratio']
    data['volume_weighted_acceleration'] = data['acceleration'] * data['volume_divergence']
    
    # Range Efficiency Calculation
    data['true_range_1'] = data['high'] - data['low']
    data['true_range_2'] = (data['high'] - data['close'].shift(1)).abs()
    data['true_range_3'] = (data['low'] - data['close'].shift(1)).abs()
    data['true_range'] = data[['true_range_1', 'true_range_2', 'true_range_3']].max(axis=1)
    data['efficiency_ratio'] = (data['close'] - data['open']).abs() / data['true_range']
    
    # Gap Analysis Enhancement
    data['overnight_gap'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['abs_gap_strength'] = data['overnight_gap'].abs()
    data['gap_momentum_interaction'] = data['acceleration'] * np.sign(data['overnight_gap'])
    
    # Volatility and Amplitude Context
    data['daily_return'] = data['close'].pct_change()
    data['volatility_20d'] = data['daily_return'].rolling(20).std()
    data['volatility_percentile'] = data['volatility_20d'].rolling(20).apply(lambda x: (x.iloc[-1] > np.percentile(x, 80)) if len(x) == 20 else np.nan, raw=False)
    data['daily_amplitude'] = (data['high'] - data['low']) / data['close']
    data['amplitude_percentile'] = data['daily_amplitude'].rolling(20).apply(lambda x: (x.iloc[-1] > np.percentile(x, 80)) if len(x) == 20 else np.nan, raw=False)
    
    # Volume Persistence Confirmation
    data['volume_direction'] = np.where(data['volume'] > data['volume'].shift(1), 1, 
                                       np.where(data['volume'] < data['volume'].shift(1), -1, 0))
    data['volume_persistence'] = data['volume_direction'].rolling(3).sum()
    
    # Breakout Condition Integration
    data['volume_20d_avg'] = data['volume'].rolling(20).mean()
    data['volume_breakout'] = (data['volume'] > 1.5 * data['volume_20d_avg']).astype(int)
    data['amplitude_20d_avg'] = data['daily_amplitude'].rolling(20).mean()
    data['range_breakout'] = (data['daily_amplitude'] > 1.5 * data['amplitude_20d_avg']).astype(int)
    data['combined_breakout'] = data['volume_breakout'] & data['range_breakout']
    
    # Divergence Pattern Synthesis
    data['efficiency_weighted_acceleration'] = data['volume_weighted_acceleration'] * data['efficiency_ratio']
    data['gap_enhanced_divergence'] = data['efficiency_weighted_acceleration'] * data['gap_momentum_interaction']
    data['volume_persistence_divergence'] = data['gap_enhanced_divergence'] * data['volume_persistence']
    
    # Regime-Adaptive Enhancement
    data['volatility_scaling'] = np.where(data['volatility_percentile'] == 1, 1.5, 1.0)
    data['amplitude_scaling'] = np.where(data['amplitude_percentile'] == 1, 1.3, 1.0)
    data['regime_enhanced'] = data['volume_persistence_divergence'] * data['volatility_scaling'] * data['amplitude_scaling']
    
    # Breakout Confirmation Layer
    data['breakout_multiplier'] = np.where(data['combined_breakout'] == 1, 2.0,
                                          np.where(data['volume_breakout'] == 1, 1.5,
                                                  np.where(data['range_breakout'] == 1, 1.3, 1.0)))
    
    # Final Alpha Signal
    data['core_alpha'] = data['regime_enhanced'] * data['breakout_multiplier']
    
    # Signal strength classification based on conditions
    conditions = [
        (data['acceleration'] > 0) & (data['efficiency_ratio'] > 0.7) & (data['volume_persistence'] > 1) & 
        (data['overnight_gap'] > 0) & (data['combined_breakout'] == 1),  # Strong Bullish
        (data['acceleration'] > 0) & (data['efficiency_ratio'] > 0.4) & (data['volume_persistence'] > 0) & 
        (data['overnight_gap'] > 0),  # Moderate Bullish
        (data['acceleration'] < 0) & (data['efficiency_ratio'] > 0.7) & (data['volume_persistence'] < -1) & 
        (data['overnight_gap'] < 0) & (data['combined_breakout'] == 1),  # Strong Bearish
        (data['acceleration'] < 0) & (data['efficiency_ratio'] > 0.4) & (data['volume_persistence'] < 0) & 
        (data['overnight_gap'] < 0),  # Moderate Bearish
    ]
    
    choices = [2.0, 1.5, -2.0, -1.5]
    data['signal_strength'] = np.select(conditions, choices, default=1.0)
    
    # Final alpha factor
    data['alpha_factor'] = data['core_alpha'] * data['signal_strength']
    
    return data['alpha_factor']
