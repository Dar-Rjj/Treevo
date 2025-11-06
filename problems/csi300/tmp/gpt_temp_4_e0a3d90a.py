import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Asymmetric Volatility Fractals
    data['Upward_Fractal'] = (data['high'] - np.maximum(data['open'], data['close'])) / data['close'].shift(1) * data['volume']
    data['Downward_Fractal'] = (np.minimum(data['open'], data['close']) - data['low']) / data['close'].shift(1) * data['volume']
    data['Fractal_Asymmetry'] = data['Upward_Fractal'] - data['Downward_Fractal']
    
    # Asymmetry Persistence
    def calc_persistence(series):
        if len(series) < 4:
            return np.nan
        signs = np.sign(series.iloc[-4:-1])  # t-3, t-2, t-1
        current_sign = np.sign(series.iloc[-1])
        persistence_count = np.sum(signs == current_sign)
        return persistence_count / 3
    
    data['Asymmetry_Persistence'] = data['Fractal_Asymmetry'].rolling(window=4, min_periods=4).apply(calc_persistence, raw=False)
    
    # Multi-Timeframe Fractal Patterns
    data['Short_term_Fractal'] = data['Fractal_Asymmetry'] * data['volume']
    data['Medium_term_Fractal'] = (data['Fractal_Asymmetry'] - data['Fractal_Asymmetry'].shift(3)) * data['volume']
    data['Fractal_Divergence'] = data['Short_term_Fractal'] - data['Medium_term_Fractal']
    data['Fractal_Momentum'] = data['Fractal_Divergence'] - data['Fractal_Divergence'].shift(1)
    
    # Fractal Breakout Detection
    data['Volatility_Breakout'] = (data['high'] - data['high'].shift(1)) * (data['low'].shift(1) - data['low']) * data['volume']
    
    high_rolling_max = data['high'].rolling(window=3, min_periods=3).max().shift(1)
    data['Fractal_Break_Intensity'] = (data['high'] - high_rolling_max) / (data['high'] - data['low'])
    
    vol_avg = (data['high'].shift(1) - data['low'].shift(1) + 
               data['high'].shift(2) - data['low'].shift(2) + 
               data['high'].shift(3) - data['low'].shift(3)) / 3
    data['Breakout_Alignment'] = ((data['high'] - data['high'].shift(1)) / 
                                 (data['high'].shift(1) - data['low'].shift(1)) * 
                                 np.sign((data['high'] - data['low']) - vol_avg))
    
    # Memory-Enhanced Price Dynamics
    data['Recent_High_Memory'] = data['high'].rolling(window=3, min_periods=3).max().shift(1)
    data['Recent_Low_Memory'] = data['low'].rolling(window=3, min_periods=3).min().shift(1)
    data['Memory_Break_Intensity'] = (data['high'] - data['Recent_High_Memory']) / (data['high'] - data['low'])
    data['Memory_Hold_Intensity'] = (data['Recent_Low_Memory'] - data['low']) / (data['high'] - data['low'])
    
    # Volume-Enhanced Fractal Reactions
    data['Break_Volume_Confirmation'] = data['volume'] * data['Memory_Break_Intensity']
    data['Hold_Volume_Confirmation'] = data['volume'] * data['Memory_Hold_Intensity']
    data['Fractal_Confirmation_Asymmetry'] = data['Break_Volume_Confirmation'] - data['Hold_Volume_Confirmation']
    data['Confirmation_Momentum'] = data['Fractal_Confirmation_Asymmetry'] - data['Fractal_Confirmation_Asymmetry'].shift(1)
    
    # Multi-scale Memory Effects
    data['Short_term_Memory_Impact'] = data['Memory_Break_Intensity'] * data['volume']
    data['Medium_term_Memory_Impact'] = (data['Memory_Break_Intensity'] - data['Memory_Break_Intensity'].shift(3)) * data['volume']
    data['Memory_Impact_Divergence'] = data['Short_term_Memory_Impact'] - data['Medium_term_Memory_Impact']
    
    # Memory Persistence
    data['Memory_Persistence'] = data['Memory_Break_Intensity'].rolling(window=4, min_periods=4).apply(calc_persistence, raw=False)
    
    # Fractal Momentum Construction
    data['Opening_Fractal'] = (np.abs(data['open'] - data['close'].shift(1)) / 
                              (data['high'].shift(1) - data['low'].shift(1)) * data['volume'])
    data['Closing_Momentum'] = (np.abs(data['close'] - data['open']) / 
                               (data['high'] - data['low']) * (data['close'] - data['open']))
    
    # Range Persistence
    def range_persistence(series):
        if len(series) < 2:
            return 0
        count = 0
        for i in range(len(series)-1, 0, -1):
            if series.iloc[i] > series.iloc[i-1]:
                count += 1
            else:
                break
        return count
    
    data['Range_Persistence'] = (data['high'] - data['low']).rolling(window=5, min_periods=2).apply(range_persistence, raw=False)
    
    # Volatility-Price Flow
    def volatility_regime_count(high_low_series, close_open_series):
        if len(high_low_series) < 2:
            return 0
        count = 0
        current_regime = np.sign(close_open_series.iloc[-1])
        for i in range(len(high_low_series)-1, 0, -1):
            if np.sign(close_open_series.iloc[i]) == current_regime:
                count += 1
            else:
                break
        return count
    
    data['Volatility_Price_Flow'] = data.apply(
        lambda x: volatility_regime_count(
            pd.Series([x.name] * len(data), index=data.index), 
            data['close'] - data['open']
        ) * (x['close'] - x['open']), axis=1
    )
    
    # Asymmetric Momentum Patterns
    data['Fractal_Asymmetry_Momentum'] = data['Fractal_Momentum'] * data['volume']
    data['Memory_Confirmation_Momentum'] = data['Confirmation_Momentum'] * data['Fractal_Confirmation_Asymmetry']
    
    data['Volume_Fractal_Coherence'] = (np.sign(data['close']/data['close'].shift(1) - 1) * 
                                       np.sign(data['volume']/data['volume'].shift(1) - 1))
    data['Asymmetry_Fractal_Alignment'] = data['Fractal_Asymmetry'] * data['Volume_Fractal_Coherence']
    
    # Multi-Timeframe Convergence
    data['Volatility_Convergence'] = ((data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5)) * 
                                     (data['close'] - data['open']) / (data['close'].shift(1) - data['open'].shift(1)))
    data['Fractal_Divergence_MTF'] = ((data['high'] - data['low']) - (data['high'].shift(5) - data['low'].shift(5))) * \
                                    np.sign(data['close'] - data['open']) * data['volume']
    data['Memory_Fractal_Integration'] = data['Memory_Impact_Divergence'] * data['Fractal_Divergence_MTF']
    data['Convergence_Momentum'] = data['Volatility_Convergence'] - data['Volatility_Convergence'].shift(1)
    
    # Regime-Dependent Signal Synthesis
    vol_avg_3 = (data['high'].shift(1) - data['low'].shift(1) + 
                 data['high'].shift(2) - data['low'].shift(2) + 
                 data['high'].shift(3) - data['low'].shift(3)) / 3
    data['Regime_Momentum'] = (np.sign((data['high'] - data['low']) - vol_avg_3) * 
                              ((data['high'] - data['low']) - (data['high'].shift(5) - data['low'].shift(5))))
    
    data['Fractal_Breakout_Signal'] = data['Volatility_Breakout'] * data['Breakout_Alignment']
    data['Memory_Fractal_Coherence'] = data['Fractal_Asymmetry_Momentum'] * data['Memory_Confirmation_Momentum']
    data['Volume_Fractal_Integration'] = data['Volume_Fractal_Coherence'] * data['Fractal_Divergence_MTF']
    
    # Regime-Specific Multipliers
    high_fractal_boost = np.where(np.abs(data['Fractal_Asymmetry']) > 0.02, 1.5, 1.0)
    low_fractal_reduction = np.where(np.abs(data['Fractal_Asymmetry']) < 0.005, 0.6, 1.0)
    volume_surge_enhancement = np.where(data['volume'] > 1.8 * data['volume'].shift(1), 1.4, 1.0)
    volume_drop_adjustment = np.where(data['volume'] < 0.5 * data['volume'].shift(1), 0.5, 1.0)
    
    # Coherence-Validated Factors
    primary_factor = data['Memory_Fractal_Coherence'] * data['Memory_Persistence']
    secondary_factor = data['Volume_Fractal_Integration'] * data['Asymmetry_Persistence']
    tertiary_factor = data['Fractal_Breakout_Signal'] * data['Range_Persistence']
    quaternary_factor = data['Regime_Momentum'] * data['Volatility_Price_Flow']
    
    # Apply regime multipliers
    primary_factor *= high_fractal_boost * volume_surge_enhancement
    secondary_factor *= high_fractal_boost * volume_surge_enhancement
    tertiary_factor *= low_fractal_reduction * volume_drop_adjustment
    quaternary_factor *= low_fractal_reduction * volume_drop_adjustment
    
    # Signal Quality Assessment
    data['Fractal_Signal_Quality'] = np.abs(data['Fractal_Asymmetry_Momentum']) * data['volume']
    data['Memory_Signal_Quality'] = np.abs(data['Memory_Confirmation_Momentum']) * data['Fractal_Confirmation_Asymmetry']
    data['Convergence_Signal_Quality'] = np.abs(data['Convergence_Momentum']) * data['Volatility_Convergence']
    data['Integration_Signal_Quality'] = np.abs(data['Memory_Fractal_Integration']) * data['Volume_Fractal_Coherence']
    
    # Adaptive Weighting Mechanism
    fractal_adaptive_weights = np.abs(data['Fractal_Asymmetry'])
    volume_adaptive_weights = data['volume'] / data['volume'].shift(1)
    persistence_adaptive_weights = data['Asymmetry_Persistence']
    
    # Final Dynamic Alpha Calculation
    total_quality = (data['Fractal_Signal_Quality'] + data['Memory_Signal_Quality'] + 
                    data['Convergence_Signal_Quality'] + data['Integration_Signal_Quality'])
    
    # Weighted combination
    weighted_primary = primary_factor * fractal_adaptive_weights
    weighted_secondary = secondary_factor * volume_adaptive_weights
    weighted_tertiary = tertiary_factor * persistence_adaptive_weights
    weighted_quaternary = quaternary_factor * volume_adaptive_weights
    
    # Final alpha with quality normalization
    final_alpha = (weighted_primary + weighted_secondary + weighted_tertiary + weighted_quaternary)
    
    # Normalize by total signal quality to reduce noise
    alpha_factor = final_alpha / (total_quality + 1e-8)
    
    return alpha_factor
