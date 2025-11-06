import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Regime Classification
    data['daily_range'] = data['high'] - data['low']
    data['avg_range_5'] = data['daily_range'].rolling(window=5).mean()
    data['median_range_20'] = data['daily_range'].rolling(window=20).median()
    
    # Regime classification
    high_vol_condition = data['avg_range_5'] > (1.3 * data['median_range_20'])
    low_vol_condition = data['avg_range_5'] < (0.8 * data['median_range_20'])
    data['vol_regime'] = 'normal'
    data.loc[high_vol_condition, 'vol_regime'] = 'high'
    data.loc[low_vol_condition, 'vol_regime'] = 'low'
    
    # Regime-Adaptive Momentum Components
    def get_regime_window(regime):
        if regime == 'high':
            return 3
        elif regime == 'low':
            return 8
        else:
            return 5
    
    # Calculate momentum based on regime
    momentum_values = []
    for i in range(len(data)):
        if i < 8:  # Need at least 8 days for low volatility regime
            momentum_values.append(np.nan)
            continue
            
        regime = data['vol_regime'].iloc[i]
        window = get_regime_window(regime)
        
        if i < window:
            momentum_values.append(np.nan)
        else:
            close_current = data['close'].iloc[i]
            close_prev = data['close'].iloc[i - window]
            momentum = (close_current - close_prev) / close_prev
            momentum_values.append(momentum)
    
    data['momentum'] = momentum_values
    
    # Momentum acceleration
    data['momentum_accel'] = data['momentum'] - data['momentum'].shift(1)
    
    # Fractal Efficiency Enhancement
    efficiency_values = []
    for i in range(len(data)):
        if i < 8:
            efficiency_values.append(np.nan)
            continue
            
        regime = data['vol_regime'].iloc[i]
        window = get_regime_window(regime)
        
        if i < window:
            efficiency_values.append(np.nan)
        else:
            abs_change = abs(data['close'].iloc[i] - data['close'].iloc[i - window])
            
            # Calculate cumulative movement
            cum_movement = 0
            for j in range(window):
                cum_movement += abs(data['close'].iloc[i - j] - data['close'].iloc[i - j - 1])
            
            efficiency_ratio = abs_change / cum_movement if cum_movement != 0 else 0
            efficiency_values.append(efficiency_ratio)
    
    data['efficiency_ratio'] = efficiency_values
    data['enhanced_momentum'] = data['momentum'] * data['efficiency_ratio']
    
    # Volume Confirmation Framework
    # Volume-Range Alignment
    data['volume_range_corr'] = data['volume'].rolling(window=5).corr(data['daily_range'])
    data['alignment_score'] = data['volume_range_corr'] * np.sign(data['momentum'])
    
    # Volume-Price Divergence
    data['vwap_8'] = (data['close'] * data['volume']).rolling(window=8).sum() / data['volume'].rolling(window=8).sum()
    data['price_slope'] = (data['close'] - data['close'].shift(8)) / 8
    data['vwap_slope'] = (data['vwap_8'] - data['vwap_8'].shift(8)) / 8
    data['divergence_score'] = np.sign(data['price_slope']) * (data['price_slope'] - data['vwap_slope'])
    
    # Volume Absorption Dynamics
    data['mid_price'] = (data['high'] + data['low']) / 2
    data['upward_absorption'] = np.where(data['close'] > data['mid_price'], data['volume'], 0)
    data['downward_absorption'] = np.where(data['close'] < data['mid_price'], data['volume'], 0)
    
    upward_sum = data['upward_absorption'].rolling(window=5).sum()
    downward_sum = data['downward_absorption'].rolling(window=5).sum()
    data['absorption_asymmetry'] = (upward_sum - downward_sum) / (upward_sum + downward_sum + 1e-8)
    data['absorption_momentum'] = data['absorption_asymmetry'] - data['absorption_asymmetry'].shift(5)
    
    # Pressure Asymmetry Integration
    # Directional Pressure Analysis
    data['opening_pressure'] = (data['open'] - data['close'].shift(1)) * data['volume']
    data['closing_pressure'] = (data['close'] - data['open']) * data['volume']
    data['net_pressure'] = data['closing_pressure'] - data['opening_pressure']
    
    # Pressure Momentum
    data['upside_pressure'] = np.maximum(0, data['net_pressure']).rolling(window=5).mean()
    data['downside_pressure'] = np.maximum(0, -data['net_pressure']).rolling(window=5).mean()
    data['pressure_momentum'] = np.log(1 + data['upside_pressure']) - np.log(1 + data['downside_pressure'])
    
    # Pressure-Volume Confirmation
    data['pressure_volume_alignment'] = data['net_pressure'].rolling(window=5).corr(data['volume'])
    data['confirmed_pressure'] = data['pressure_momentum'] * data['pressure_volume_alignment']
    
    # Signal Integration & Persistence
    # Primary Signal Components
    data['momentum_core'] = data['enhanced_momentum'] * data['absorption_momentum']
    data['volume_confirmation'] = data['alignment_score'] * (1 - abs(data['divergence_score']))
    data['pressure_component'] = data['confirmed_pressure'] * data['efficiency_ratio']
    
    # Momentum Persistence Filter
    data['momentum_consistency'] = data['momentum'].rolling(window=3).corr(data['momentum'].shift(1))
    data['persistence_score'] = np.maximum(0, data['momentum_consistency'])
    
    # Regime-Specific Integration
    def calculate_regime_alpha(row):
        if pd.isna(row['momentum_core']) or pd.isna(row['persistence_score']):
            return np.nan
            
        if row['vol_regime'] == 'high':
            primary = row['momentum_core'] * row['volume_confirmation']
            secondary = row['pressure_component'] * row['persistence_score']
            return primary * 0.7 + secondary * 0.3
        elif row['vol_regime'] == 'low':
            primary = row['momentum_core'] * row['pressure_component']
            secondary = row['volume_confirmation'] * row['persistence_score']
            return primary * 0.6 + secondary * 0.4
        else:  # normal
            balanced = row['momentum_core'] * row['volume_confirmation']
            momentum = row['pressure_component'] * row['persistence_score']
            return balanced * 0.5 + momentum * 0.5
    
    data['regime_weighted_alpha'] = data.apply(calculate_regime_alpha, axis=1)
    
    # Final Alpha Output
    data['final_alpha'] = data['regime_weighted_alpha'] * data['persistence_score']
    
    return data['final_alpha']
