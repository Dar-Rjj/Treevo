import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Ensure data is sorted by date
    df = df.sort_index()
    
    # Initialize result series
    result = pd.Series(index=df.index, dtype=float)
    
    # Calculate all required components
    for i in range(5, len(df)):
        # Fractal Momentum Construction
        if i >= 3:
            asymmetric_momentum = ((df.iloc[i]['close'] / df.iloc[i-1]['close'] - 1) * 
                                 (df.iloc[i]['volume'] / df.iloc[i-1]['volume'] - 1) * 
                                 (df.iloc[i]['high'] - df.iloc[i]['low']) / 
                                 max(df.iloc[i-2]['high'] - df.iloc[i-2]['low'], 1e-6))
            
            cross_association_momentum = ((df.iloc[i]['close'] - df.iloc[i-3]['close']) * 
                                        (df.iloc[i]['volume'] - df.iloc[i-3]['volume']))
            
            momentum_acceleration = (asymmetric_momentum - 
                                   ((df.iloc[i]['close'] / df.iloc[i-2]['close'] - 1) * 
                                    (df.iloc[i]['volume'] / df.iloc[i-2]['volume'] - 1) * 
                                    (df.iloc[i]['high'] - df.iloc[i]['low']) / 
                                    max(df.iloc[i-3]['high'] - df.iloc[i-3]['low'], 1e-6)))
        else:
            asymmetric_momentum = cross_association_momentum = momentum_acceleration = 0
        
        # Microstructure Efficiency Components
        range_efficiency = (df.iloc[i]['volume'] / 
                          max(df.iloc[i]['high'] - df.iloc[i]['low'], 
                              abs(df.iloc[i]['high'] - df.iloc[i-1]['close']), 
                              abs(df.iloc[i]['low'] - df.iloc[i-1]['close']), 1e-6))
        
        if i >= 2:
            intraday_fractal_efficiency = (abs((df.iloc[i]['high'] + df.iloc[i]['low'])/2 - 
                                             (df.iloc[i]['open'] + df.iloc[i]['close'])/2) / 
                                         max(df.iloc[i]['high'] - df.iloc[i]['low'], 1e-6) * 
                                         (df.iloc[i]['close'] / df.iloc[i-1]['close'] - 1) * 
                                         (df.iloc[i]['volume'] / max(df.iloc[i-2]['volume'], 1e-6)))
        else:
            intraday_fractal_efficiency = 0
        
        closing_pressure = ((df.iloc[i]['close'] - df.iloc[i]['low']) / 
                          max(df.iloc[i]['high'] - df.iloc[i]['low'], 1e-6) - 
                          (df.iloc[i]['high'] - df.iloc[i]['close']) / 
                          max(df.iloc[i]['high'] - df.iloc[i]['low'], 1e-6))
        
        # Volatility-Volume Fractal Alignment
        if i >= 1:
            volatility_adjusted_return = ((df.iloc[i]['close'] / df.iloc[i-1]['close'] - 1) / 
                                        max(df.iloc[i]['high'] - df.iloc[i]['low'], 1e-6) / 
                                        max(df.iloc[i]['close'], 1e-6))
            
            if i >= 2:
                volume_distribution_fractal = ((df.iloc[i]['volume'] / max(df.iloc[i-2]['volume'], 1e-6)) * 
                                             (df.iloc[i]['close'] / df.iloc[i-1]['close'] - 1) * 
                                             (df.iloc[i]['amount'] / max(df.iloc[i-1]['amount'], 1e-6) - 1))
            else:
                volume_distribution_fractal = 0
            
            if i >= 1:
                volume_volatility_coherence = (np.sign(df.iloc[i]['volume'] / max(df.iloc[i-1]['volume'], 1e-6) - 1) * 
                                             np.sign((df.iloc[i]['high'] - df.iloc[i]['low']) / max(df.iloc[i]['close'], 1e-6) - 
                                                     (df.iloc[i-1]['high'] - df.iloc[i-1]['low']) / max(df.iloc[i-1]['close'], 1e-6)))
            else:
                volume_volatility_coherence = 0
        else:
            volatility_adjusted_return = volume_distribution_fractal = volume_volatility_coherence = 0
        
        # Multi-Scale Fractal Integration
        micro_fractal = ((df.iloc[i]['close'] - df.iloc[i]['open']) / 
                        max(df.iloc[i]['high'] - df.iloc[i]['low'], 1e-6) * 
                        (df.iloc[i]['volume'] / max(df.iloc[i-2]['volume'], 1e-6) - 1) * 
                        (df.iloc[i]['amount'] / max(df.iloc[i-1]['amount'], 1e-6) - 1))
        
        if i >= 3:
            high_range_3 = df.iloc[i-3:i+1]['high'].max() - df.iloc[i-3:i+1]['low'].min()
            meso_fractal = ((df.iloc[i]['close'] - df.iloc[i-3]['close']) / 
                           max(high_range_3, 1e-6) * 
                           (df.iloc[i]['volume'] / max(df.iloc[i-3]['volume'], 1e-6) - 1) * 
                           (df.iloc[i]['amount'] / max(df.iloc[i-3]['amount'], 1e-6) - 1))
        else:
            meso_fractal = 0
        
        if i >= 5:
            high_range_5 = df.iloc[i-5:i+1]['high'].max() - df.iloc[i-5:i+1]['low'].min()
            macro_fractal = ((df.iloc[i]['close'] - df.iloc[i-5]['close']) / 
                            max(high_range_5, 1e-6) * 
                            (df.iloc[i]['volume'] / max(df.iloc[i-5]['volume'], 1e-6) - 1) * 
                            (df.iloc[i]['amount'] / max(df.iloc[i-5]['amount'], 1e-6) - 1))
        else:
            macro_fractal = 0
        
        # Fractal Regime Detection
        efficiency_regime = int(range_efficiency > 
                              abs((df.iloc[i]['high'] + df.iloc[i]['low'])/2 - 
                                  (df.iloc[i]['open'] + df.iloc[i]['close'])/2) / 
                              max(df.iloc[i]['high'] - df.iloc[i]['low'], 1e-6))
        
        if i >= 1:
            volume_regime = int((df.iloc[i]['volume'] / max(df.iloc[i-1]['volume'], 1e-6) > 1) and 
                              ((df.iloc[i]['amount'] / max(df.iloc[i]['volume'], 1e-6)) - 
                               (df.iloc[i-1]['amount'] / max(df.iloc[i-1]['volume'], 1e-6)) > 0))
        else:
            volume_regime = 0
        
        if i >= 1:
            volatility_regime = int((df.iloc[i]['high'] - df.iloc[i]['low']) / max(df.iloc[i]['close'], 1e-6) > 
                                  (df.iloc[i-1]['high'] - df.iloc[i-1]['low']) / max(df.iloc[i-1]['close'], 1e-6))
        else:
            volatility_regime = 0
        
        # Signal Construction
        core_fractal_momentum = momentum_acceleration * cross_association_momentum * intraday_fractal_efficiency
        volatility_enhanced = core_fractal_momentum * volatility_adjusted_return * volume_volatility_coherence
        regime_adapted = volatility_enhanced * (efficiency_regime + volume_regime + volatility_regime)
        
        # Multi-Timeframe Fractal Weighting
        efficiency_dominance = (range_efficiency / 
                              max(abs((df.iloc[i]['high'] + df.iloc[i]['low'])/2 - 
                                   (df.iloc[i]['open'] + df.iloc[i]['close'])/2) / 
                                  max(df.iloc[i]['high'] - df.iloc[i]['low'], 1e-6), 1e-6))
        
        if i >= 1:
            volatility_dominance = (volatility_adjusted_return / 
                                  max(abs((df.iloc[i]['high'] - df.iloc[i]['low'])/max(df.iloc[i]['close'], 1e-6) - 
                                       (df.iloc[i-1]['high'] - df.iloc[i-1]['low'])/max(df.iloc[i-1]['close'], 1e-6)), 1e-6))
        else:
            volatility_dominance = 0
        
        volume_dominance = volume_distribution_fractal / max(range_efficiency, 1e-6)
        
        # Final Alpha Construction
        multi_scale_momentum = regime_adapted * micro_fractal * meso_fractal * macro_fractal
        weight_adjusted_signal = multi_scale_momentum * (efficiency_dominance + volatility_dominance + volume_dominance)
        
        if i >= 2:
            final_alpha = (weight_adjusted_signal * closing_pressure * 
                          (df.iloc[i]['volume'] / max(df.iloc[i-2]['volume'], 1e-6)))
        else:
            final_alpha = 0
        
        result.iloc[i] = final_alpha
    
    # Fill initial values with 0
    result = result.fillna(0)
    
    return result
