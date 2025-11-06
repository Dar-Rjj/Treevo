import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['fractal_efficiency_stress'] = (np.abs(data['close'] - data['open']) / 
                                        (data['high'] - data['low'])) * \
                                       ((data['high'] - data['low']) / 
                                        np.abs(data['close'] - data['close'].shift(1)))
    
    data['gap_fractal_stress'] = (np.abs(data['open'] - data['close'].shift(1)) / 
                                 (data['high'] - data['low'])) * \
                                (data['volume'] / (data['high'] - data['low']))
    
    data['volume_fractal_stress'] = (data['volume'] / (data['high'] - data['low'])) * \
                                   (data['volume'] / data['volume'].shift(1))
    
    # Fractal Rejection Patterns
    data['upper_fractal_rejection'] = ((data['high'] - np.maximum(data['open'], data['close'])) / 
                                      (data['high'] - data['low'])) * data['volume']
    
    data['lower_fractal_rejection'] = ((np.minimum(data['open'], data['close']) - data['low']) / 
                                      (data['high'] - data['low'])) * data['volume']
    
    data['net_fractal_rejection'] = data['upper_fractal_rejection'] - data['lower_fractal_rejection']
    
    # Fractal Momentum Construction
    data['clean_fractal_momentum'] = ((data['close'] / data['close'].shift(1) - 1) * 
                                     (np.abs(data['close'] - data['open']) / (data['high'] - data['low'])))
    
    data['short_term_fractal_acceleration'] = (
        ((data['close'] - data['close'].shift(3)) / data['close'].shift(3) - 
         (data['close'].shift(3) - data['close'].shift(6)) / data['close'].shift(6)) * 
        (np.log(data['high'] - data['low']) / np.log(data['close'] - data['open'] + 1))
    )
    
    # Fractal Divergence Signals
    data['efficiency_stress_divergence'] = np.sign(data['fractal_efficiency_stress']) * \
                                          np.sign(data['fractal_efficiency_stress'].shift(1))
    
    data['rejection_stress_alignment'] = np.sign(data['net_fractal_rejection']) * \
                                        np.sign(data['net_fractal_rejection'].shift(1))
    
    data['volume_efficiency_divergence'] = np.sign(data['volume_fractal_stress']) * \
                                          np.sign(data['fractal_efficiency_stress'])
    
    # Fractal Persistence Validation
    def calculate_persistence(series, window=3):
        persistence = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window:
                signs = [np.sign(series.iloc[i-j]) for j in range(window)]
                matches = sum(1 for j in range(1, window) if signs[j] == signs[j-1] and not np.isnan(signs[j]) and not np.isnan(signs[j-1]))
                persistence.iloc[i] = matches / (window - 1)
        return persistence
    
    data['efficiency_fractal_persistence'] = calculate_persistence(data['fractal_efficiency_stress'])
    data['momentum_fractal_consistency'] = calculate_persistence(data['clean_fractal_momentum'])
    data['volume_fractal_flow'] = calculate_persistence(data['volume'].diff())
    
    # Final Alpha Construction
    primary_factor = (data['clean_fractal_momentum'] * 
                     data['fractal_efficiency_stress'] * 
                     data['efficiency_stress_divergence'])
    
    secondary_factor = (data['net_fractal_rejection'] * 
                       data['volume_fractal_stress'] * 
                       data['volume_fractal_flow'])
    
    tertiary_factor = (data['short_term_fractal_acceleration'] * 
                      data['gap_fractal_stress'] * 
                      data['momentum_fractal_consistency'])
    
    # Combine factors with weights
    final_alpha = (0.5 * primary_factor + 
                   0.3 * secondary_factor + 
                   0.2 * tertiary_factor)
    
    return final_alpha
