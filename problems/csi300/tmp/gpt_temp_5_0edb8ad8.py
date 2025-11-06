import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Volatility Regime Detection
    data['prev_close'] = data['close'].shift(1)
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['prev_close']),
            np.abs(data['low'] - data['prev_close'])
        )
    )
    data['vol_regime'] = data['true_range'] > data['true_range'].rolling(window=10, min_periods=1).median()
    
    # Momentum Structure
    data['intraday_momentum_up'] = (data['high'] - data['close']) / data['close']
    data['intraday_momentum_down'] = (data['close'] - data['low']) / data['close']
    data['intraday_momentum'] = data['intraday_momentum_up'] - data['intraday_momentum_down']
    
    data['price_change'] = data['close'] - data['prev_close']
    data['momentum_persistence'] = np.sign(data['price_change']) * (data['price_change'] / data['prev_close'])
    
    # Volume Dynamics
    data['volume_ma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
    data['volume_trend'] = data['volume'] / data['volume_ma_5']
    data['volume_price_sync'] = np.sign(data['price_change']) * data['volume_trend']
    
    # Order Flow Pressure
    data['price_up'] = data['close'] > data['prev_close']
    data['price_down'] = data['close'] < data['prev_close']
    
    # Calculate up-tick and down-tick volumes for rolling 5-day window
    up_tick_volumes = []
    down_tick_volumes = []
    
    for i in range(len(data)):
        start_idx = max(0, i - 4)
        window_data = data.iloc[start_idx:i+1]
        up_vol = window_data.loc[window_data['price_up'], 'volume'].sum()
        down_vol = window_data.loc[window_data['price_down'], 'volume'].sum()
        up_tick_volumes.append(up_vol)
        down_tick_volumes.append(down_vol)
    
    data['up_tick_volume'] = up_tick_volumes
    data['down_tick_volume'] = down_tick_volumes
    data['order_flow_pressure'] = data['up_tick_volume'] - data['down_tick_volume']
    
    # Core Factor
    data['core_factor'] = data['intraday_momentum'] * data['volume_price_sync'] * data['order_flow_pressure']
    
    # Regime Enhancement
    data['regime_enhanced'] = np.where(
        data['vol_regime'],
        data['core_factor'] * data['momentum_persistence'],  # High Vol regime
        data['core_factor'] * data['volume_trend']           # Low Vol regime
    )
    
    # Final Alpha
    alpha = data['regime_enhanced']
    
    return alpha
