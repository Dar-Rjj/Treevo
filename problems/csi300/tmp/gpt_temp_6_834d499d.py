import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Elasticity-Entropy Momentum Dynamics factor
    """
    data = df.copy()
    
    # Elastic Momentum Structure
    # Multi-Scale Momentum Elasticity
    data['short_term_elasticity'] = (data['close'] - data['close'].shift(2)) / (data['close'].shift(2) - data['close'].shift(4) + 1e-8)
    data['medium_term_elasticity'] = (data['close'] - data['close'].shift(5)) / (data['close'].shift(5) - data['close'].shift(10) + 1e-8)
    data['elasticity_divergence'] = data['short_term_elasticity'] - data['medium_term_elasticity']
    
    # Nonlinear Momentum Acceleration
    data['quadratic_momentum'] = (data['close'] / data['close'].shift(2))**2 - (data['close'] / data['close'].shift(1))**2
    data['second_derivative_acceleration'] = (data['close'] / data['close'].shift(1)) - (data['close'].shift(1) / data['close'].shift(2))
    
    # Momentum Persistence Structure
    momentum_persistence = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 5:
            count = sum([data['close'].iloc[i] > data['close'].iloc[i-j] for j in range(1, 6)])
            momentum_persistence.iloc[i] = count / 5
        else:
            momentum_persistence.iloc[i] = 0
    data['momentum_persistence'] = momentum_persistence
    data['elastic_persistence'] = data['momentum_persistence'] * data['elasticity_divergence']
    
    # Pressure-Entropy Dynamics
    # Bidirectional Pressure Calculation
    data['buy_side_pressure'] = (data['high'] - data['open']) * data['volume']
    data['sell_side_pressure'] = (data['open'] - data['low']) * data['volume']
    data['net_pressure_ratio'] = (data['buy_side_pressure'] - data['sell_side_pressure']) / (data['buy_side_pressure'] + data['sell_side_pressure'] + 1e-8)
    
    # Price-Volume Entropy Integration
    price_volume_entropy = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 4:
            window_data = data.iloc[i-4:i+1]
            avg_price = window_data['close'].mean()
            total_volume = window_data['volume'].sum()
            if total_volume > 0:
                entropy = -sum((window_data['close'] - avg_price)**2 * window_data['volume']) / total_volume
            else:
                entropy = 0
            price_volume_entropy.iloc[i] = entropy
        else:
            price_volume_entropy.iloc[i] = 0
    data['price_volume_entropy'] = price_volume_entropy
    data['entropy_pressure_alignment'] = data['net_pressure_ratio'] * data['price_volume_entropy']
    
    # Flow-Entropy Asymmetry
    data['entropy_enhanced_intraday_flow'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) * data['price_volume_entropy']
    data['volume_entropy_collapse'] = (((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8)) - 
                                     ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8))) * data['price_volume_entropy']
    
    # Volatility-Elasticity Regime Classification
    # Fractal Volatility Components
    data['price_path_volatility'] = (abs(data['high'] - data['close']) + abs(data['close'] - data['low'])) / (data['high'] - data['low'] + 1e-8)
    data['volume_volatility'] = data['volume'] / (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3) + 1e-8)
    
    # Multi-Scale Elasticity Dynamics
    short_term_vol = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if i >= 2:
            vol = np.sqrt(sum([(data['close'].iloc[j] / data['close'].iloc[j-1] - 1)**2 for j in range(i-2, i+1)]))
            short_term_vol.iloc[i] = vol
        else:
            short_term_vol.iloc[i] = 0
    data['short_term_volatility'] = short_term_vol
    
    data['volume_weighted_price_dispersion'] = (data['high'] - data['low']) * data['volume'] / (data['amount'] + 1e-8)
    
    data['elastic_range_momentum'] = (((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1) + 1e-8)) - 
                                    ((data['high'].shift(1) - data['low'].shift(1)) / (data['high'].shift(2) - data['low'].shift(2) + 1e-8))) * data['price_volume_entropy']
    
    # Elastic Asymmetry Structure
    # Multi-Scale Asymmetry Alignment
    data['micro_macro_elasticity_coherence'] = data['short_term_elasticity'] / (data['medium_term_elasticity'] + 1e-8)
    
    # Entropy-Elasticity Correlation
    data['volume_weighted_acceleration'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) * data['quadratic_momentum']
    data['entropy_elasticity_correlation'] = data['price_volume_entropy'] * data['elasticity_divergence']
    
    # Gap-Entropy Range Dynamics
    data['entropy_gap_absorption'] = ((data['close'] - data['open']) / (data['open'] - data['close'].shift(1) + 1e-8)) * data['price_volume_entropy']
    data['range_compression'] = (data['high'] - data['low']) / (data['close'].shift(1) + 1e-8)
    
    # Order Flow Breakout Detection
    # Asymmetric Breakout Components
    data['price_breakout'] = (data['close'] > data['high'].shift(1)) & (data['close'] > data['high'].shift(2)) & (data['close'] > data['high'].shift(3))
    data['volume_surge'] = data['volume'] > 1.5 * (data['volume'].shift(1) + data['volume'].shift(2) + data['volume'].shift(3)) / 3
    data['breakout_asymmetry'] = (data['buy_side_pressure'] - data['sell_side_pressure']) / (data['high'] - data['low'] + 1e-8)
    
    # Elastic Breakout Analysis
    data['compression_breakout_interaction'] = data['range_compression'] * data['breakout_asymmetry']
    data['elastic_breakout_momentum'] = data['breakout_asymmetry'] * data['elasticity_divergence']
    
    # Final Composite Alpha Factor
    # Regime Detection and Signal Selection
    volatility_threshold = data['price_path_volatility'].rolling(window=20, min_periods=10).quantile(0.7)
    
    regime_signals = pd.Series(index=data.index, dtype=float)
    for i in range(len(data)):
        if data['price_breakout'].iloc[i] and data['volume_surge'].iloc[i]:
            # Breakout regime
            regime_signals.iloc[i] = data['elastic_breakout_momentum'].iloc[i] * data['entropy_pressure_alignment'].iloc[i]
        elif data['price_path_volatility'].iloc[i] > volatility_threshold.iloc[i]:
            # High volatility regime
            regime_signals.iloc[i] = data['elasticity_divergence'].iloc[i] * data['volume_weighted_price_dispersion'].iloc[i]
        elif data['price_path_volatility'].iloc[i] < volatility_threshold.iloc[i] * 0.5:
            # Low volatility regime
            regime_signals.iloc[i] = data['second_derivative_acceleration'].iloc[i] * data['range_compression'].iloc[i]
        else:
            # Default regime
            regime_signals.iloc[i] = data['micro_macro_elasticity_coherence'].iloc[i] * data['price_volume_entropy'].iloc[i]
    
    # Elastic-Entropy Synthesis
    core_momentum = regime_signals
    entropy_confirmation = core_momentum * data['price_volume_entropy']
    final_factor = entropy_confirmation * data['elastic_persistence']
    
    return final_factor
