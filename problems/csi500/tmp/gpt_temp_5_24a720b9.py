import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Momentum Diffusion Components
    data['price_momentum'] = data['close'] / data['close'].shift(1) - 1
    data['volume_momentum'] = data['volume'] / data['volume'].shift(1) - 1
    data['range_momentum'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1)) - 1
    
    # Microstructure Rejection Signals
    data['intraday_rejection'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    # Efficiency asymmetry with safe division
    high_open_diff = data['high'] - data['open']
    open_low_diff = data['open'] - data['low']
    data['efficiency_asymmetry'] = np.where(high_open_diff != 0, (data['close'] - data['open']) / high_open_diff, 0) - \
                                  np.where(open_low_diff != 0, (data['open'] - data['close']) / open_low_diff, 0)
    
    data['close_to_mid_range'] = data['close'] / ((data['high'] + data['low']) / 2) - 1
    
    # Momentum-Persistence Integration
    # Price streak count
    price_changes = data['close'] - data['close'].shift(1)
    price_signs = np.sign(price_changes)
    price_streak = price_signs.rolling(window=5, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and x.iloc[i] != 0]), 
        raw=False
    )
    
    # Volume streak count
    volume_changes = data['volume'] - data['volume'].shift(1)
    volume_signs = np.sign(volume_changes)
    volume_streak = volume_signs.rolling(window=5, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == x.iloc[i-1] and x.iloc[i] != 0]), 
        raw=False
    )
    
    # Rejection persistence
    intraday_sign = np.sign(data['close'] - data['open'])
    data['rejection_persistence'] = (intraday_sign * data['intraday_rejection']).rolling(window=3, min_periods=1).sum()
    
    # Asymmetry Components
    data['volatility_asymmetry'] = ((data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))) * \
                                  np.sign(data['open'] - data['close'].shift(1))
    
    volume_median_5d = data['volume'].rolling(window=5, min_periods=1).median()
    data['volume_pressure'] = (data['volume'] / volume_median_5d) * np.sign(data['close'] - data['open'])
    
    data['momentum_divergence'] = np.abs((data['close'] - data['close'].shift(5)) - (data['close'] - data['close'].shift(20)))
    
    # Factor Synthesis
    momentum_rejection_convergence = data['price_momentum'] * data['intraday_rejection'] * data['close_to_mid_range']
    asymmetry_weighted_momentum = data['efficiency_asymmetry'] * data['volatility_asymmetry'] * data['price_momentum']
    persistence_pressure_factor = (price_streak + volume_streak) * data['volume_pressure'] * data['rejection_persistence']
    divergence_corrected_factor = data['momentum_divergence'] * data['range_momentum'] * data['volume_momentum']
    
    # Final factor combination
    final_factor = (momentum_rejection_convergence + 
                   asymmetry_weighted_momentum + 
                   persistence_pressure_factor - 
                   divergence_corrected_factor)
    
    return final_factor
