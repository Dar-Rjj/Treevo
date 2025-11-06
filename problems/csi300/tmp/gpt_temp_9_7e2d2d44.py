import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Elastic Momentum Structure
    # Multi-Scale Momentum Elasticity
    data['short_term_elasticity'] = (data['close'] - data['close'].shift(2)) / (data['close'].shift(2) - data['close'].shift(4))
    data['medium_term_elasticity'] = (data['close'] - data['close'].shift(5)) / (data['close'].shift(5) - data['close'].shift(10))
    data['elasticity_divergence'] = data['short_term_elasticity'] - data['medium_term_elasticity']
    
    # Momentum Persistence Structure
    momentum_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 5:
            current_close = data['close'].iloc[i]
            count = sum(1 for j in range(1, 6) if current_close > data['close'].iloc[i-j])
            momentum_persistence.iloc[i] = count / 5
        else:
            momentum_persistence.iloc[i] = 0
    data['momentum_persistence'] = momentum_persistence
    data['elastic_persistence'] = data['momentum_persistence'] * data['elasticity_divergence']
    
    # Pressure-Entropy Dynamics
    # Bidirectional Pressure Calculation
    data['buy_side_pressure'] = (data['high'] - data['open']) * data['volume']
    data['sell_side_pressure'] = (data['open'] - data['low']) * data['volume']
    data['net_pressure_ratio'] = (data['buy_side_pressure'] - data['sell_side_pressure']) / (data['buy_side_pressure'] + data['sell_side_pressure'] + 1e-10)
    
    # Price-Volume Entropy Integration
    price_volume_entropy = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            avg_price = window_data['close'].mean()
            total_volume = window_data['volume'].sum()
            if total_volume > 0:
                entropy_sum = sum((window_data['close'].iloc[j] - avg_price) ** 2 * window_data['volume'].iloc[j] 
                                for j in range(len(window_data)))
                price_volume_entropy.iloc[i] = -entropy_sum / total_volume
            else:
                price_volume_entropy.iloc[i] = 0
        else:
            price_volume_entropy.iloc[i] = 0
    data['price_volume_entropy'] = price_volume_entropy
    data['entropy_pressure_alignment'] = data['net_pressure_ratio'] * data['price_volume_entropy']
    
    # Volatility-Elasticity Regime Classification
    # Fractal Volatility Components
    data['price_path_volatility'] = (abs(data['high'] - data['close']) + abs(data['close'] - data['low'])) / (data['high'] - data['low'] + 1e-10)
    
    volume_volatility = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 3:
            volume_volatility.iloc[i] = data['volume'].iloc[i] / (data['volume'].iloc[i-1] + data['volume'].iloc[i-2] + data['volume'].iloc[i-3] + 1e-10)
        else:
            volume_volatility.iloc[i] = 0
    data['volume_volatility'] = volume_volatility
    
    # Multi-Scale Elasticity Dynamics
    short_term_volatility = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 2:
            vol_sum = sum((data['close'].iloc[j] / data['close'].iloc[j-1] - 1) ** 2 for j in range(i-2, i+1))
            short_term_volatility.iloc[i] = np.sqrt(vol_sum)
        else:
            short_term_volatility.iloc[i] = 0
    data['short_term_volatility'] = short_term_volatility
    
    data['volume_weighted_price_dispersion'] = (data['high'] - data['low']) * data['volume'] / (data['amount'] + 1e-10)
    
    # Order Flow Breakout Detection
    # Asymmetric Breakout Components
    price_breakout = pd.Series(index=data.index, dtype=bool)
    volume_surge = pd.Series(index=data.index, dtype=bool)
    
    for i in range(len(data)):
        if i >= 3:
            price_breakout.iloc[i] = data['close'].iloc[i] > max(data['high'].iloc[i-1], data['high'].iloc[i-2], data['high'].iloc[i-3])
            avg_volume = (data['volume'].iloc[i-1] + data['volume'].iloc[i-2] + data['volume'].iloc[i-3]) / 3
            volume_surge.iloc[i] = data['volume'].iloc[i] > 1.5 * avg_volume
        else:
            price_breakout.iloc[i] = False
            volume_surge.iloc[i] = False
    
    data['price_breakout'] = price_breakout
    data['volume_surge'] = volume_surge
    data['breakout_asymmetry'] = (data['buy_side_pressure'] - data['sell_side_pressure']) / (data['high'] - data['low'] + 1e-10)
    data['elastic_breakout_momentum'] = data['breakout_asymmetry'] * data['elasticity_divergence']
    
    # Final Composite Alpha Factor
    # Regime Detection and Signal Selection
    volatility_threshold = data['price_path_volatility'].rolling(window=20, min_periods=1).mean()
    
    factor = pd.Series(index=data.index, dtype=float)
    
    for i in range(len(data)):
        if data['price_breakout'].iloc[i] and data['volume_surge'].iloc[i]:
            # Breakout regime
            factor.iloc[i] = data['elastic_breakout_momentum'].iloc[i] * data['entropy_pressure_alignment'].iloc[i]
        elif data['price_path_volatility'].iloc[i] > volatility_threshold.iloc[i]:
            # High volatility regime
            factor.iloc[i] = data['elasticity_divergence'].iloc[i] * data['volume_weighted_price_dispersion'].iloc[i]
        elif data['price_path_volatility'].iloc[i] < volatility_threshold.iloc[i] * 0.5:
            # Low volatility regime
            factor.iloc[i] = data['elastic_persistence'].iloc[i]
        else:
            # Default regime
            factor.iloc[i] = data['elasticity_divergence'].iloc[i] * data['price_volume_entropy'].iloc[i]
    
    # Elastic-Entropy Synthesis
    final_factor = factor * data['price_volume_entropy']
    
    return final_factor
