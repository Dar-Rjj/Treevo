import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Gap Dynamics
    data['gap_magnitude'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_fill_efficiency'] = (data['close'] - data['open']) / (data['close'].shift(1) - data['open'])
    data['gap_fill_efficiency'] = data['gap_fill_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Gap Momentum Enhancement
    data['gap_persistence'] = np.where(
        (data['gap_magnitude'] > 0) & (data['close'] > data['open']), 1,
        np.where((data['gap_magnitude'] < 0) & (data['close'] < data['open']), -1, 0)
    )
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['gap_momentum'] = data['gap_magnitude'] * data['intraday_strength']
    
    # Breakout Patterns
    data['up_breakout'] = np.where(
        (data['high'] > data['high'].shift(1)) & (data['gap_magnitude'] > 0), 1, 0
    )
    data['down_breakout'] = np.where(
        (data['low'] < data['low'].shift(1)) & (data['gap_magnitude'] < 0), -1, 0
    )
    
    # Range Efficiency Analysis
    data['short_term_range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['short_term_range_efficiency'] = data['short_term_range_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    data['medium_term_range_efficiency'] = data['short_term_range_efficiency'].rolling(window=5, min_periods=3).mean()
    data['range_efficiency_momentum'] = data['short_term_range_efficiency'].diff(3)
    
    # Volume Dynamics
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=3).mean()
    data['volume_clustering'] = data['volume'] / data['volume_5d_avg']
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(1)) - 1
    
    # Trade Size Analysis
    data['large_trade_ratio'] = data['amount'] / (data['volume'] / 100)
    data['large_trade_ratio'] = data['large_trade_ratio'].replace([np.inf, -np.inf], np.nan)
    
    # Large Trade Confirmation (simplified as percentile rank)
    data['large_trade_confirmation'] = data['large_trade_ratio'].rolling(window=20, min_periods=10).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 90)) if len(x.dropna()) > 0 else np.nan
    )
    data['large_trade_confirmation'] = np.where(data['large_trade_confirmation'] == 1, 1.2, 1.0)
    
    # Composite Alpha Generation
    data['gap_breakout_alignment'] = np.where(
        data['up_breakout'] == 1, 1, np.where(data['down_breakout'] == -1, -1, 0)
    )
    
    data['momentum_range_divergence'] = data['gap_momentum'] - data['range_efficiency_momentum']
    
    data['volume_enhanced_signal'] = (
        data['gap_breakout_alignment'] * data['momentum_range_divergence'] * data['volume_acceleration']
    )
    
    # Final Quantum Factor
    data['quantum_factor'] = (
        data['volume_enhanced_signal'] * 
        data['large_trade_confirmation'] * 
        (1 - data['short_term_range_efficiency'])
    )
    
    # Clean and return the factor
    factor = data['quantum_factor'].replace([np.inf, -np.inf], np.nan)
    return factor
