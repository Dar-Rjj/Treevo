import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Efficiency
    # Ultra-Short (2-day) Efficiency
    data['eff_2d'] = np.abs(data['close'] - data['close'].shift(2)) / (
        data['high'].rolling(window=2, min_periods=2).max() - 
        data['low'].rolling(window=2, min_periods=2).min()
    )
    
    # Short-Term (5-day) Efficiency
    data['eff_5d'] = np.abs(data['close'] - data['close'].shift(5)) / (
        data['high'].rolling(window=5, min_periods=5).max() - 
        data['low'].rolling(window=5, min_periods=5).min()
    )
    
    # 20-day Efficiency for decay calculation
    data['eff_20d'] = np.abs(data['close'] - data['close'].shift(20)) / (
        data['high'].rolling(window=20, min_periods=20).max() - 
        data['low'].rolling(window=20, min_periods=20).min()
    )
    
    # Efficiency Decay
    data['efficiency_decay'] = (data['eff_5d'] - data['eff_2d']) * (data['eff_20d'] - data['eff_5d'])
    
    # Microstructure Fracture
    # Opening Efficiency
    data['opening_efficiency'] = np.abs(data['close'] - data['open']) / np.abs(data['open'] - data['close'].shift(1))
    data['opening_efficiency'] = data['opening_efficiency'].replace([np.inf, -np.inf], np.nan)
    
    # Intraday Pressure
    high_low_range = data['high'] - data['low']
    high_diff = np.abs(data['high'] - data['high'].shift(1))
    low_diff = np.abs(data['low'] - data['low'].shift(1))
    data['intraday_pressure'] = (np.abs(data['close'] - data['close'].shift(1)) / high_low_range) - (
        np.minimum(high_diff, low_diff) / high_low_range
    )
    
    # Closing Momentum
    hl_midpoint = (data['high'] + data['low']) / 2
    close_open_diff = np.abs(data['close'] - data['open'])
    close_close_diff = np.abs(data['close'] - data['close'].shift(1))
    data['closing_momentum'] = ((data['close'] - hl_midpoint) / hl_midpoint) * (
        close_open_diff / close_close_diff.replace(0, np.nan)
    )
    
    # Volatility-Momentum Integration
    # Volatility Skew
    data['volatility_skew'] = ((data['high'] - data['open']) / high_low_range) - (
        (data['open'] - data['low']) / high_low_range
    )
    
    # Momentum Decay
    close_3d_diff = data['close'] - data['close'].shift(3)
    close_3_6_diff = np.abs(data['close'].shift(3) - data['close'].shift(6))
    close_8d_diff = data['close'] - data['close'].shift(8)
    close_8_16_diff = np.abs(data['close'].shift(8) - data['close'].shift(16))
    
    data['momentum_decay'] = (close_3d_diff / close_3_6_diff.replace(0, np.nan)) - (
        close_8d_diff / close_8_16_diff.replace(0, np.nan)
    )
    
    # Range Compression
    current_range = data['high'] - data['low']
    prev_range = data['high'].shift(4) - data['low'].shift(4)
    range_ratio = current_range / prev_range.replace(0, np.nan)
    
    # Count range compression events in rolling window
    compression_count = (range_ratio < 0.8).rolling(window=5, min_periods=5).sum()
    data['range_compression'] = compression_count * range_ratio
    
    # Volume Dynamics
    # Volume Asymmetry
    up_volume = data['volume'].rolling(window=5, min_periods=5).apply(
        lambda x: np.sum(x[np.array(data['close'] > data['open']).astype(bool)[-len(x):]])
    )
    down_volume = data['volume'].rolling(window=5, min_periods=5).apply(
        lambda x: np.sum(x[np.array(data['close'] < data['open']).astype(bool)[-len(x):]])
    )
    data['volume_asymmetry'] = up_volume / down_volume.replace(0, np.nan)
    
    # Volume Pressure
    data['volume_pressure'] = (data['volume'] / data['volume'].shift(1)) - (
        data['volume'] / ((data['volume'].shift(2) + data['volume'].shift(1)) / 2)
    )
    
    # Amount Efficiency
    vwap_current = data['amount'] / (data['volume'] * data['close']).replace(0, np.nan)
    amount_avg = (data['amount'].shift(2) + data['amount'].shift(1)) / 2
    data['amount_efficiency'] = vwap_current * (data['amount'] / amount_avg.replace(0, np.nan))
    
    # Composite Alpha
    # Core Signal
    data['core_signal'] = data['efficiency_decay'] * data['opening_efficiency'] * data['volume_asymmetry']
    
    # Volatility Factor
    data['volatility_factor'] = (
        data['volatility_skew'] * 0.25 + 
        data['momentum_decay'] * 0.20 - 
        data['range_compression'] * 0.15
    )
    
    # Microstructure Factor
    data['microstructure_factor'] = (
        data['intraday_pressure'] * 0.12 + 
        data['closing_momentum'] * 0.10
    )
    
    # Volume Factor
    data['volume_factor'] = (
        data['volume_pressure'] * 0.15 + 
        data['amount_efficiency'] * 0.05
    )
    
    # Final Alpha
    data['alpha'] = data['core_signal'] * (
        data['volatility_factor'] + 
        data['microstructure_factor'] + 
        data['volume_factor']
    )
    
    return data['alpha']
