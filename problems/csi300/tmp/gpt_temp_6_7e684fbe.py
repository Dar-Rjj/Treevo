import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Momentum
    data['momentum_3d'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['momentum_5d'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['momentum_10d'] = (data['close'] - data['close'].shift(10)) / data['close'].shift(10)
    
    # Momentum Acceleration
    data['short_acceleration'] = (data['momentum_3d'] - data['momentum_5d']) / 2
    data['medium_acceleration'] = (data['momentum_5d'] - data['momentum_10d']) / 5
    data['acceleration_strength'] = data['momentum_5d'] * (data['short_acceleration'] + data['medium_acceleration']) / 2
    
    # Intraday Pressure Dynamics
    data['buying_pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    data['selling_pressure'] = ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)) * data['volume']
    data['net_pressure'] = data['buying_pressure'] - data['selling_pressure']
    
    # Price-Volume Alignment
    volume_diff = data['volume'] - data['volume'].shift(3)
    data['price_volume_alignment'] = np.sign(data['momentum_3d']) * np.sign(volume_diff) * np.abs(data['momentum_3d'] * volume_diff)
    data['pressure_signal'] = data['price_volume_alignment'] * np.sign(data['net_pressure'])
    
    # Volatility Breakout System
    data['rolling_max_high'] = data['high'].rolling(window=5, min_periods=1).max()
    data['rolling_min_low'] = data['low'].rolling(window=5, min_periods=1).min()
    
    data['upper_breakout'] = (data['close'] - data['rolling_max_high']) * np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['lower_breakout'] = (data['close'] - data['rolling_min_low']) * (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['breakout_score'] = data['upper_breakout'] + data['lower_breakout']
    
    # Volatility Enhancement
    data['daily_range'] = (data['high'] - data['low']) / ((data['high'] + data['low']) / 2 + 1e-8)
    data['volatility_breakout'] = data['breakout_score'] / (data['daily_range'] + 1e-8)
    
    # Flow Persistence Patterns
    # Consecutive Direction
    price_change = data['close'] - data['close'].shift(1)
    sign_changes = price_change.rolling(window=5).apply(lambda x: len(set(np.sign(x.dropna()))) if len(x.dropna()) == 5 else np.nan, raw=False)
    data['consecutive_direction'] = 5 - sign_changes + 1
    
    # Volume Correlation
    data['volume_correlation'] = data['volume'].rolling(window=5).corr(data['volume'].shift(1))
    
    # Persistence Score
    data['flow_consistency'] = data['consecutive_direction'] * data['volume_correlation']
    data['persistence_momentum'] = data['flow_consistency'] * (1 - np.abs(data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8))
    
    # Composite Alpha Generation
    data['core_factor'] = data['acceleration_strength'] * data['pressure_signal'] * data['volatility_breakout']
    data['flow_enhanced'] = data['core_factor'] * data['persistence_momentum']
    
    # Volume normalization
    data['volume_avg_10d'] = data['volume'].rolling(window=10, min_periods=1).mean()
    data['final_alpha'] = data['flow_enhanced'] * (data['volume'] / (data['volume_avg_10d'] + 1e-8))
    
    return data['final_alpha']
