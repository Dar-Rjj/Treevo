import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum Efficiency Acceleration with Microstructure Divergence factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-period Momentum Efficiency Analysis
    # Short-term momentum efficiency (t-5 to t)
    data['momentum_5d'] = data['close'] / data['close'].shift(5) - 1
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / (data['high'] - data['low']).replace(0, np.nan)
    data['efficiency_momentum'] = data['intraday_efficiency'] / data['intraday_efficiency'].shift(1) - 1
    
    # Medium-term momentum efficiency (t-10 to t)
    data['momentum_10d'] = data['close'] / data['close'].shift(10) - 1
    
    # Calculate rolling efficiency average for medium-term
    efficiency_rolling = []
    for i in range(len(data)):
        if i >= 9:
            window_data = data.iloc[i-9:i+1]
            avg_efficiency = (abs(window_data['close'] - window_data['open']) / 
                            (window_data['high'] - window_data['low']).replace(0, np.nan)).mean()
        else:
            avg_efficiency = np.nan
        efficiency_rolling.append(avg_efficiency)
    
    data['efficiency_10d_avg'] = efficiency_rolling
    data['efficiency_trend'] = data['intraday_efficiency'] / data['intraday_efficiency'].rolling(5).mean()
    
    # Momentum acceleration detection
    data['momentum_acceleration_ratio'] = (data['momentum_5d'] / data['momentum_10d']).replace([np.inf, -np.inf], np.nan)
    data['price_acceleration'] = data['momentum_5d'] - (data['close'] / data['close'].shift(1) - 1)
    
    # Volume-Microstructure Divergence Assessment
    data['volume_momentum_5d'] = data['volume'] / data['volume'].shift(5) - 1
    data['volume_momentum_change'] = data['volume_momentum_5d'] - data['volume_momentum_5d'].shift(1)
    data['volume_acceleration'] = (data['volume'] / data['volume'].shift(5) - 1) / (data['volume'] / data['volume'].shift(10) - 1).replace(0, np.nan)
    
    # Price-volume divergence strength
    data['divergence_5d'] = data['momentum_5d'] - data['volume_momentum_5d']
    data['divergence_10d'] = data['momentum_10d'] - (data['volume'] / data['volume'].shift(10) - 1)
    data['divergence_consistency'] = np.sign(data['divergence_5d']) * np.sign(data['divergence_10d'])
    
    # Microstructure stability confirmation
    data['daily_range'] = data['high'] - data['low']
    data['range_stability'] = data['daily_range'] / data['daily_range'].rolling(5).mean()
    
    # Volume concentration analysis
    data['volume_spike_ratio'] = data['volume'] / data['volume'].rolling(4).mean().shift(1)
    volume_rolling_stats = data['volume'].rolling(5)
    data['volume_stability'] = 1 / (1 + volume_rolling_stats.std() / volume_rolling_stats.mean())
    data['volume_efficiency'] = data['amount'] / data['volume'].replace(0, np.nan)
    
    # Composite Signal Generation
    # Momentum efficiency acceleration score
    data['base_momentum'] = data['momentum_5d'] * data['intraday_efficiency']
    data['acceleration_multiplier'] = np.tanh(data['momentum_acceleration_ratio'])
    data['efficiency_adjustment'] = data['efficiency_momentum'] * data['range_stability']
    
    # Volume divergence adjustment
    divergence_boost = np.where(data['divergence_5d'] > 0, 1.2, 1.0)
    divergence_penalty = np.where(data['divergence_5d'] < 0, 0.8, 1.0)
    data['divergence_adjustment'] = divergence_boost * divergence_penalty
    
    # Volume quality filtering
    volume_quality = np.where((data['volume_spike_ratio'] > 0.5) & (data['volume_spike_ratio'] < 2.0), 
                             data['volume_stability'], 0)
    
    # Volatility and efficiency integration
    # Calculate 20-day ATR
    def calculate_atr(data_window):
        high_low = data_window['high'] - data_window['low']
        high_close_prev = abs(data_window['high'] - data_window['close'].shift(1))
        low_close_prev = abs(data_window['low'] - data_window['close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range.mean()
    
    atr_values = []
    for i in range(len(data)):
        if i >= 19:
            window_data = data.iloc[i-19:i+1]
            atr = calculate_atr(window_data)
        else:
            atr = np.nan
        atr_values.append(atr)
    
    data['atr_20d'] = atr_values
    
    # Final composite factor calculation
    momentum_component = (data['base_momentum'] * 
                         data['acceleration_multiplier'] * 
                         (1 + data['efficiency_adjustment']))
    
    volume_component = (data['divergence_adjustment'] * 
                       volume_quality * 
                       data['divergence_consistency'])
    
    # Primary factor
    primary_factor = momentum_component * volume_component
    
    # Risk-adjusted final factor
    final_factor = primary_factor / data['atr_20d'].replace(0, np.nan)
    
    # Clean up infinite values and handle NaN
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan)
    
    return final_factor
