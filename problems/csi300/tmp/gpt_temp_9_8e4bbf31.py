import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily price range
    data['price_range'] = (data['high'] - data['low']) / data['close']
    
    # Volatility regime classification
    data['volatility_median'] = data['price_range'].rolling(window=20, min_periods=1).median()
    data['high_vol_regime'] = (data['price_range'] > 1.5 * data['volatility_median']).astype(int)
    
    # Volume regime classification
    data['volume_median'] = data['volume'].rolling(window=20, min_periods=1).median()
    data['high_volume_regime'] = (data['volume'] > 2 * data['volume_median']).astype(int)
    
    # Core Price-Volume Dynamics
    data['gap_momentum'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['intraday_strength'] = (data['close'] - data['open']) / data['open']
    
    # Volume-Price Divergence
    price_change = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    volume_change = data['volume'] / data['volume'].shift(1)
    data['volume_price_divergence'] = np.sign(data['close'] - data['close'].shift(1)) * (volume_change - np.abs(price_change))
    
    # Order Flow Imbalance (Net Pressure)
    def calculate_net_pressure(window):
        if len(window) < 3:
            return np.nan
        up_volume = window[window['close'] > window['open']]['volume'].sum()
        down_volume = window[window['close'] < window['open']]['volume'].sum()
        total_volume = window['volume'].sum()
        return (up_volume - down_volume) / total_volume if total_volume > 0 else 0
    
    # Calculate rolling net pressure
    net_pressure_values = []
    for i in range(len(data)):
        if i >= 2:
            window_data = data.iloc[i-2:i+1]
            net_pressure_values.append(calculate_net_pressure(window_data))
        else:
            net_pressure_values.append(np.nan)
    
    data['net_pressure'] = net_pressure_values
    
    # Price Efficiency (Efficiency Ratio)
    def calculate_efficiency_ratio(window):
        if len(window) < 5:
            return np.nan
        net_change = window['close'].iloc[-1] - window['close'].iloc[0]
        total_variation = np.sum(np.abs(window['close'].diff().dropna()))
        return net_change / total_variation if total_variation != 0 else 0
    
    # Calculate rolling efficiency ratio
    efficiency_values = []
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            efficiency_values.append(calculate_efficiency_ratio(window_data))
        else:
            efficiency_values.append(np.nan)
    
    data['efficiency_ratio'] = efficiency_values
    
    # Base Factor
    data['base_factor'] = data['gap_momentum'] * data['volume_price_divergence'] * data['net_pressure']
    
    # Regime-Enhanced Factor
    data['regime_enhanced_factor'] = np.nan
    
    # High Volatility regime
    high_vol_mask = data['high_vol_regime'] == 1
    data.loc[high_vol_mask, 'regime_enhanced_factor'] = (
        data.loc[high_vol_mask, 'base_factor'] * 
        data.loc[high_vol_mask, 'intraday_strength'] * 
        (1 + data.loc[high_vol_mask, 'efficiency_ratio'])
    )
    
    # Normal regime
    normal_mask = data['high_vol_regime'] == 0
    data.loc[normal_mask, 'regime_enhanced_factor'] = (
        data.loc[normal_mask, 'base_factor'] * 
        data.loc[normal_mask, 'efficiency_ratio'] * 
        (1 + np.abs(data.loc[normal_mask, 'net_pressure']))
    )
    
    # Final Alpha
    data['final_alpha'] = data['regime_enhanced_factor'] * np.sign(data['base_factor'])
    
    # Fill NaN values with 0
    data['final_alpha'] = data['final_alpha'].fillna(0)
    
    return data['final_alpha']
