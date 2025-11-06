import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-Scale Volatility-Pressure Asymmetry with Fractal Efficiency Fusion
    Generates regime-adaptive alpha factor combining volatility asymmetry, pressure efficiency, and fractal dynamics
    """
    data = df.copy()
    
    # Ensure data is sorted by date
    data = data.sort_index()
    
    # Basic calculations
    data['prev_close'] = data['close'].shift(1)
    data['range'] = (data['high'] - data['low']) / data['close']
    data['buying_efficiency'] = (data['close'] - data['low']) / (data['high'] - data['low']).replace(0, np.nan)
    data['selling_efficiency'] = (data['high'] - data['close']) / (data['high'] - data['low']).replace(0, np.nan)
    
    # Volatility asymmetry analysis
    data['up_day'] = data['close'] > data['prev_close']
    data['down_day'] = data['close'] < data['prev_close']
    
    # 5-day volatility accumulations
    for window in [5]:
        data[f'up_vol_{window}d'] = data['range'].rolling(window).apply(
            lambda x: x[data['up_day'].iloc[-window:].values].sum() if len(x) == window else np.nan, 
            raw=False
        )
        data[f'down_vol_{window}d'] = data['range'].rolling(window).apply(
            lambda x: x[data['down_day'].iloc[-window:].values].sum() if len(x) == window else np.nan, 
            raw=False
        )
    
    data['vol_asymmetry_5d'] = data['up_vol_5d'] / data['down_vol_5d'].replace(0, np.nan)
    
    # Pressure efficiency analysis
    data['buying_pressure'] = data['buying_efficiency'] * data['volume']
    data['selling_pressure'] = data['selling_efficiency'] * data['volume']
    
    for window in [5]:
        data[f'buying_eff_{window}d'] = data['buying_efficiency'].rolling(window).sum()
        data[f'selling_eff_{window}d'] = data['selling_efficiency'].rolling(window).sum()
        data[f'net_pressure_{window}d'] = (data['buying_pressure'] - data['selling_pressure']).rolling(window).sum()
    
    # Volume concentration
    data['volume_5d_avg'] = data['volume'].rolling(5).mean()
    data['volume_concentration'] = data['volume'] / data['volume_5d_avg'].replace(0, np.nan)
    
    # Fractal Efficiency Analysis
    data['close_change_abs'] = abs(data['close'] - data['close'].shift(1))
    
    # Short-term efficiency (3-day)
    data['st_efficiency_numerator'] = abs(data['close'] - data['close'].shift(3))
    data['st_efficiency_denominator'] = data['close_change_abs'].rolling(3).sum()
    data['short_term_efficiency'] = data['st_efficiency_numerator'] / data['st_efficiency_denominator'].replace(0, np.nan)
    
    # Medium-term efficiency (8-day)
    data['mt_efficiency_numerator'] = abs(data['close'] - data['close'].shift(8))
    data['mt_efficiency_denominator'] = data['close_change_abs'].rolling(8).sum()
    data['medium_term_efficiency'] = data['mt_efficiency_numerator'] / data['mt_efficiency_denominator'].replace(0, np.nan)
    
    data['efficiency_momentum'] = data['short_term_efficiency'] - data['medium_term_efficiency']
    
    # Fractal volume dynamics
    data['fractal_volume_st'] = data['volume'] / data['volume'].rolling(5).min().replace(0, np.nan)
    data['fractal_volume_mt'] = data['volume'] / data['volume'].rolling(10).min().replace(0, np.nan)
    data['fractal_consistency'] = data['fractal_volume_st'] / data['fractal_volume_mt'].replace(0, np.nan)
    
    # Regime classification
    data['high_efficiency'] = data['short_term_efficiency'] > data['short_term_efficiency'].rolling(20).quantile(0.6)
    data['low_fractal'] = data['fractal_consistency'] < data['fractal_consistency'].rolling(20).quantile(0.4)
    data['low_efficiency'] = data['short_term_efficiency'] < data['short_term_efficiency'].rolling(20).quantile(0.4)
    data['high_fractal'] = data['fractal_consistency'] > data['fractal_consistency'].rolling(20).quantile(0.6)
    
    # Regime flags
    data['momentum_regime'] = data['high_efficiency'] & data['low_fractal']
    data['reversion_regime'] = data['low_efficiency'] & data['high_fractal']
    data['transition_regime'] = ~(data['momentum_regime'] | data['reversion_regime'])
    
    # Asymmetric momentum factors
    data['volatility_momentum'] = ((data['high'] - data['close']) / data['close'] - 
                                  (data['high'].shift(5) - data['close'].shift(5)) / data['close'].shift(5))
    
    data['pressure_weighted_momentum'] = (data['close'] / data['prev_close']) * (data['buying_efficiency'] - data['selling_efficiency'])
    data['asymmetry_accelerated_momentum'] = data['vol_asymmetry_5d'] * data['efficiency_momentum']
    
    # Asymmetric mean-reversion factors
    data['downside_vol_momentum'] = ((data['close'] - data['low']) / data['close'] - 
                                    (data['close'].shift(5) - data['low'].shift(5)) / data['close'].shift(5))
    
    data['gap_range_alignment'] = np.sign(data['open'] - data['prev_close']) * np.sign(data['close'] - data['open'])
    data['pressure_asymmetry_reversion'] = (data['selling_efficiency'] - data['buying_efficiency']) * data['vol_asymmetry_5d']
    
    # Component integration with regime weighting
    momentum_component = (0.5 * data['volatility_momentum'] + 
                         0.3 * data['pressure_weighted_momentum'] + 
                         0.2 * data['asymmetry_accelerated_momentum'])
    
    reversion_component = (0.5 * data['downside_vol_momentum'] + 
                          0.3 * data['pressure_asymmetry_reversion'] + 
                          0.2 * data['gap_range_alignment'])
    
    # Regime-adaptive factor calculation
    factor = np.zeros(len(data))
    
    # Momentum regime
    momentum_mask = data['momentum_regime'].fillna(False)
    factor[momentum_mask] = momentum_component[momentum_mask]
    
    # Reversion regime  
    reversion_mask = data['reversion_regime'].fillna(False)
    factor[reversion_mask] = reversion_component[reversion_mask]
    
    # Transition regime
    transition_mask = data['transition_regime'].fillna(False)
    factor[transition_mask] = (0.4 * momentum_component[transition_mask] + 
                              0.4 * reversion_component[transition_mask] + 
                              0.2 * data['net_pressure_5d'][transition_mask] / data['volume_5d_avg'][transition_mask].replace(0, np.nan))
    
    # Apply fractal consistency multiplier
    fractal_multiplier = 1 + 0.5 * (data['fractal_consistency'] - 1)
    factor = factor * fractal_multiplier
    
    # Adjust for pressure persistence
    pressure_persistence = data['net_pressure_5d'].rolling(3).mean() / data['volume_5d_avg'].replace(0, np.nan)
    factor = factor * (1 + 0.3 * np.tanh(pressure_persistence))
    
    # Clean and return
    factor_series = pd.Series(factor, index=data.index)
    factor_series = factor_series.replace([np.inf, -np.inf], np.nan)
    
    return factor_series
