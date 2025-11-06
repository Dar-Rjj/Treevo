import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Fractal Regime Transition Framework
    data['fractal_regime_transition'] = ((data['close'] - data['close'].shift(3)) / (data['high'].shift(3) - data['low'].shift(3))) - \
                                       ((data['close'] - data['close'].shift(6)) / (data['high'].shift(6) - data['low'].shift(6)))
    
    data['volume_intensity_regime_transition'] = (data['volume'] / (data['high'] - data['low'])) - \
                                               (data['volume'].shift(3) / (data['high'].shift(3) - data['low'].shift(3)))
    
    data['regime_transition_ratio'] = data['fractal_regime_transition'] / data['volume_intensity_regime_transition']
    
    # Volatility Transition Components
    data['intraday_volatility_transition'] = (data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2))
    data['interday_volatility_transition'] = abs(data['close'] - data['close'].shift(1)) / abs(data['close'].shift(2) - data['close'].shift(3))
    data['fractal_volatility_transition'] = data['intraday_volatility_transition'] * data['interday_volatility_transition']
    
    # Price-Volume Divergence Dynamics
    data['micro_price_divergence'] = ((data['close'] - data['close'].shift(1)) / (data['high'] - data['low'])) * data['amount']
    data['macro_price_divergence'] = ((data['close'] - data['close'].shift(2)) / (data['high'].shift(2) - data['low'].shift(2))) * data['amount']
    data['net_price_divergence'] = data['macro_price_divergence'] - data['micro_price_divergence']
    
    # Price-Volume Divergence
    data['up_price_divergence'] = data['amount'].rolling(window=3).apply(
        lambda x: (x * (data.loc[x.index, 'close'] > data.loc[x.index, 'open'])).sum() / x.sum() if x.sum() != 0 else 0, raw=False
    )
    data['down_price_divergence'] = data['amount'].rolling(window=3).apply(
        lambda x: (x * (data.loc[x.index, 'close'] < data.loc[x.index, 'open'])).sum() / x.sum() if x.sum() != 0 else 0, raw=False
    )
    data['price_volume_divergence'] = data['up_price_divergence'] - data['down_price_divergence']
    
    # Fractal Microstructure Divergence
    data['opening_fractal_divergence'] = ((data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))) * \
                                       (data['amount'] / data['amount'].shift(1))
    data['intraday_fractal_efficiency_divergence'] = ((data['close'] - data['open']) / (data['high'] - data['low'])) * data['price_volume_divergence']
    data['closing_fractal_pressure_divergence'] = ((data['close'] - data['low']) / (data['high'] - data['low'])) - \
                                                ((data['close'].shift(2) - data['low'].shift(2)) / (data['high'].shift(2) - data['low'].shift(2)))
    
    # Momentum Efficiency Divergent Analysis
    data['momentum_efficiency_2d'] = abs(data['close'] - data['close'].shift(2)) / \
                                   (abs(data['close'] - data['close'].shift(1)) + abs(data['close'].shift(1) - data['close'].shift(2)))
    data['momentum_efficiency_4d'] = abs(data['close'] - data['close'].shift(4)) / \
                                   (abs(data['close'] - data['close'].shift(1)) + abs(data['close'].shift(1) - data['close'].shift(2)) + 
                                    abs(data['close'].shift(2) - data['close'].shift(3)) + abs(data['close'].shift(3) - data['close'].shift(4)))
    data['momentum_efficiency_gradient'] = data['momentum_efficiency_4d'] - data['momentum_efficiency_2d']
    
    # Momentum Volatility Divergence
    data['upside_momentum_volatility'] = (data['high'] - data['close']) / (data['high'] - data['low'])
    data['downside_momentum_volatility'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['momentum_volatility_divergence_ratio'] = data['upside_momentum_volatility'] / data['downside_momentum_volatility']
    
    # Volatility-Momentum Divergent Regime Framework
    data['avg_volatility_3d'] = (data['high'] - data['low']).rolling(window=3).mean()
    data['median_volatility_15d'] = (data['high'] - data['low']).rolling(window=15).median()
    
    def volatility_regime(row):
        if row['avg_volatility_3d'] > 1.4 * row['median_volatility_15d']:
            return 1.4
        elif row['avg_volatility_3d'] < 0.7 * row['median_volatility_15d']:
            return 0.6
        else:
            return 1.0
    
    data['volatility_divergent_filter'] = data.apply(volatility_regime, axis=1)
    
    # Volume Divergent Multiplier
    data['avg_amount_8d'] = data['amount'].rolling(window=8).mean()
    data['volume_divergent_multiplier'] = np.where(data['amount'] > data['avg_amount_8d'], 1.5, 0.8)
    
    # Core Divergent Signal
    data['core_divergent_signal'] = data['fractal_regime_transition'] * data['price_volume_divergence'] * data['closing_fractal_pressure_divergence']
    
    # Base Divergent Alpha
    data['base_divergent_alpha'] = data['core_divergent_signal'] * data['volatility_divergent_filter'] * data['volume_divergent_multiplier']
    
    # Enhanced Divergent Alpha
    data['enhanced_divergent_alpha'] = data['base_divergent_alpha'] * data['momentum_efficiency_gradient']
    
    # Fractal Divergent Components
    data['range_divergent_alpha'] = data['fractal_regime_transition'] * data['intraday_volatility_transition']
    data['efficiency_divergent_alpha'] = data['intraday_fractal_efficiency_divergence'] * data['momentum_efficiency_gradient']
    data['pressure_divergent_alpha'] = data['price_volume_divergence'] * data['closing_fractal_pressure_divergence']
    
    # Multi-Fractal Divergent Strength
    data['range_divergent_strength'] = abs(data['fractal_regime_transition']) + abs(data['intraday_volatility_transition'])
    data['efficiency_divergent_strength'] = abs(data['intraday_fractal_efficiency_divergence']) + abs(data['momentum_efficiency_gradient'])
    data['pressure_divergent_strength'] = abs(data['price_volume_divergence']) + abs(data['closing_fractal_pressure_divergence'])
    
    # Adaptive Multi-Fractal Weights
    total_strength = data['range_divergent_strength'] + data['efficiency_divergent_strength'] + data['pressure_divergent_strength']
    data['range_divergent_weight'] = data['range_divergent_strength'] / total_strength
    data['efficiency_divergent_weight'] = data['efficiency_divergent_strength'] / total_strength
    data['pressure_divergent_weight'] = data['pressure_divergent_strength'] / total_strength
    
    # Composite Multi-Fractal Divergent Alpha
    data['composite_multi_fractal_divergent_alpha'] = (data['range_divergent_alpha'] * data['range_divergent_weight'] + 
                                                     data['efficiency_divergent_alpha'] * data['efficiency_divergent_weight'] + 
                                                     data['pressure_divergent_alpha'] * data['pressure_divergent_weight'])
    
    # Final Multi-Fractal Divergent Alpha
    data['final_multi_fractal_divergent_alpha'] = data['enhanced_divergent_alpha'] * data['composite_multi_fractal_divergent_alpha']
    
    return data['final_multi_fractal_divergent_alpha']
