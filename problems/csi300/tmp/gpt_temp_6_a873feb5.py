import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns
    data['ret'] = data['close'] / data['close'].shift(1) - 1
    
    # Multi-Timeframe Volatility Fractals
    data['vol_ratio_hierarchy'] = data['ret'].rolling(5).std() / data['ret'].rolling(20).std()
    data['range_fractal_dim'] = np.log(data['high'] - data['low']) / np.log(data['high'].shift(1) - data['low'].shift(1))
    
    vol_clustering = []
    for i in range(len(data)):
        if i >= 19:
            vol_window = data['ret'].iloc[i-19:i+1].std()
            count = 0
            for j in range(max(0, i-4), i+1):
                if abs(data['ret'].iloc[j]) > 2 * vol_window:
                    count += 1
            vol_clustering.append(count / 5 if i >= 4 else np.nan)
        else:
            vol_clustering.append(np.nan)
    data['vol_clustering'] = vol_clustering
    
    # Volume Fractal Dynamics
    data['volume_multi_scale_ratio'] = (data['volume'] / data['volume'].shift(1)) * (data['volume'].shift(1) / data['volume'].shift(3))
    
    vol_persistence = []
    for i in range(len(data)):
        if i >= 2:
            count = 0
            for j in range(i-2, i+1):
                if (data['volume'].iloc[j] > data['volume'].iloc[j-1]) and (data['volume'].iloc[j-1] > data['volume'].iloc[j-2]):
                    count += 1
            vol_persistence.append(count / 3)
        else:
            vol_persistence.append(np.nan)
    data['volume_fractal_persistence'] = vol_persistence
    
    data['volume_clustering_intensity'] = data['volume'].rolling(5).max() / data['volume'].rolling(5).min()
    
    # Price-Level Fractal Analysis
    support_resistance = []
    price_persistence = []
    for i in range(len(data)):
        if i >= 4:
            sr_count = 0
            pp_count = 0
            for j in range(i-4, i+1):
                if (data['close'].iloc[j] > data['close'].iloc[j-1]) and (data['close'].iloc[j-1] < data['close'].iloc[j-2]):
                    sr_count += 1
                if abs(data['close'].iloc[j] - data['close'].iloc[j-1]) < 0.005 * data['close'].iloc[j-1]:
                    pp_count += 1
            support_resistance.append(sr_count / 5)
            price_persistence.append(pp_count / 5)
        else:
            support_resistance.append(np.nan)
            price_persistence.append(np.nan)
    data['support_resistance_fractals'] = support_resistance
    data['price_level_persistence'] = price_persistence
    
    data['fractal_breakout_strength'] = (data['high'] - data['high'].rolling(5).max().shift(1)) / \
                                       (data['high'].rolling(5).max().shift(1) - data['low'].rolling(5).min().shift(1))
    
    # Microstructure Momentum Asymmetry
    data['opening_imbalance'] = (data['open'] - data['close'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    data['closing_pressure'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['intraday_reversal_strength'] = (data['high'] - data['close']) - (data['close'] - data['low'])
    
    imbalance_persistence = []
    for i in range(len(data)):
        if i >= 2:
            count = 0
            for j in range(i-2, i+1):
                if np.sign(data['opening_imbalance'].iloc[j]) == np.sign(data['closing_pressure'].iloc[j]):
                    count += 1
            imbalance_persistence.append(count / 3)
        else:
            imbalance_persistence.append(np.nan)
    data['imbalance_persistence'] = imbalance_persistence
    
    # Volume-Price Fractal Alignment
    data['volume_price_fractal_corr'] = np.sign(data['volume'] / data['volume'].shift(1) - 1) * np.sign(data['close'] / data['close'].shift(1) - 1)
    data['fractal_volume_momentum'] = (data['volume'] / data['volume'].shift(3))**(1/3) - (data['volume'].shift(3) / data['volume'].shift(6))**(1/3)
    data['price_volume_fractal_eff'] = (abs(data['close'] - data['open']) / (data['high'] - data['low'])) * np.log(data['volume'] / data['volume'].shift(1))
    
    vol_confirmation = []
    for i in range(len(data)):
        if i >= 2:
            count = 0
            for j in range(i-2, i+1):
                if (data['volume'].iloc[j] > data['volume'].iloc[j-1]) and (data['close'].iloc[j] > data['close'].iloc[j-1]):
                    count += 1
            vol_confirmation.append(count / 3)
        else:
            vol_confirmation.append(np.nan)
    data['multi_scale_volume_confirmation'] = vol_confirmation
    
    # Microstructure Acceleration
    data['trade_size_acceleration'] = (data['amount'] / data['volume']) / (data['amount'].shift(2) / data['volume'].shift(2)) - 1
    data['efficiency_acceleration'] = (abs(data['close'] - data['open']) / (data['high'] - data['low'])) / \
                                    (abs(data['close'].shift(2) - data['open'].shift(2)) / (data['high'].shift(2) - data['low'].shift(2))) - 1
    data['range_expansion_velocity'] = (data['high'] - data['low']) / (data['high'].shift(2) - data['low'].shift(2)) - 1
    
    momentum_coherence = []
    for i in range(len(data)):
        if i >= 1:
            count = 0
            for j in range(i-1, i+1):
                if np.sign(data['trade_size_acceleration'].iloc[j]) == np.sign(data['efficiency_acceleration'].iloc[j]):
                    count += 1
            momentum_coherence.append(count / 2)
        else:
            momentum_coherence.append(np.nan)
    data['microstructure_momentum_coherence'] = momentum_coherence
    
    # Fractal Velocity Components
    data['volatility_fractal_momentum'] = data['vol_ratio_hierarchy'] * data['range_fractal_dim']
    data['volume_fractal_velocity'] = data['fractal_volume_momentum'] * data['volume_clustering_intensity']
    data['microstructure_fractal_momentum'] = data['trade_size_acceleration'] * data['efficiency_acceleration']
    data['price_fractal_velocity'] = data['fractal_breakout_strength'] * data['support_resistance_fractals']
    
    # Fractal Alignment Signals
    data['volatility_volume_alignment'] = np.sign(data['volatility_fractal_momentum']) * np.sign(data['volume_fractal_velocity'])
    data['microstructure_price_alignment'] = np.sign(data['microstructure_fractal_momentum']) * np.sign(data['price_fractal_velocity'])
    
    # Fractal Divergence Patterns
    data['efficiency_volume_fractal_divergence'] = np.sign(data['efficiency_acceleration']) * np.sign(data['fractal_volume_momentum'])
    data['support_resistance_volume'] = np.sign(data['support_resistance_fractals']) * np.sign(data['volume_fractal_persistence'])
    
    # Core factors
    data['core_fractal_factor'] = data['volatility_fractal_momentum'] * data['volume_fractal_velocity']
    data['microstructure_fractal_factor'] = data['microstructure_fractal_momentum'] * data['efficiency_volume_fractal_divergence']
    data['price_level_fractal_factor'] = data['price_fractal_velocity'] * data['support_resistance_volume']
    
    # Final alpha synthesis with regime weighting
    alpha = (data['core_fractal_factor'] * 0.4 + 
             data['microstructure_fractal_factor'] * 0.3 + 
             data['price_level_fractal_factor'] * 0.3)
    
    return alpha
