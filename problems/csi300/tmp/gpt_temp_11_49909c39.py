import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Volatility Fractal Framework
    # Multi-Scale Volatility Fractals
    data['micro_vol'] = (data['high'] - data['low']) / (data['high'].shift(1) - data['low'].shift(1))
    data['short_vol'] = (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()) / (data['high'].shift(1) - data['low'].shift(1))
    data['medium_vol'] = (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min()) / (data['high'].shift(3) - data['low'].shift(3))
    
    # Volatility Persistence
    data['clustering'] = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))
    data['persistence_signal'] = np.sign(data['clustering'] - data['clustering'].shift(1)) * (data['high'] - data['low'])
    
    # Volatility Regime Classification
    data['high_vol_regime'] = (data['micro_vol'] > data['short_vol']) & (data['volume'] > data['volume'].shift(1))
    data['low_vol_regime'] = (data['micro_vol'] < data['short_vol']) & (data['volume'] < data['volume'].shift(1))
    data['transition_regime'] = data['medium_vol'] > ((data['micro_vol'] + data['short_vol']) / 2)
    
    # Volume Fractal Dynamics
    # Volume Momentum Components
    data['volume_momentum'] = (data['volume'] / data['volume'].shift(3)) - (data['volume'].shift(1) / data['volume'].shift(4))
    data['volume_intensity'] = data['volume'] / (data['high'] - data['low'])
    data['volume_vol_ratio'] = (data['volume'] / (data['high'] - data['low'])) - (data['volume'].shift(1) / (data['high'].shift(1) - data['low'].shift(1)))
    
    # Volume Fractal Structure
    data['volume_fractal_ratio'] = (data['volume'] / data['volume'].shift(1)) * (data['volume'].shift(1) / data['volume'].shift(2))
    data['amount_concentration'] = (data['amount'] / data['volume']) * (data['volume'] / data['volume'].shift(1))
    
    # Volume-Price Flow
    data['volume_price_flow'] = ((data['close'] - data['open']) / data['volume']) - ((data['close'].shift(1) - data['open'].shift(1)) / data['volume'].shift(1)) * (data['volume'] / data['volume'].shift(1))
    
    # Volume Regime Integration
    data['volume_response'] = (data['volume'] / data['volume'].shift(1)) * ((data['close'] - data['open']) / (data['high'] - data['low']))
    
    # Gap Volume
    gap_denom = data['high'].shift(1) - data['low'].shift(1)
    data['gap_volume'] = (np.abs(data['open'] - data['close'].shift(1)) / gap_denom.replace(0, np.nan)) * np.sign(data['close'] - data['open'])
    
    # Volume Persistence
    vol_persistence_count = pd.Series(index=data.index, dtype=float)
    for i in range(3, len(data)):
        window_data = data.iloc[i-3:i+1]
        count = ((window_data['close'] - window_data['open']) > (window_data['close'].shift(1) - window_data['open'].shift(1))).sum()
        vol_persistence_count.iloc[i] = count * (data['volume'].iloc[i] / data['volume'].iloc[i-1])
    data['volume_persistence'] = vol_persistence_count
    
    # Price Efficiency Fractals
    # Intraday Efficiency Measures
    intraday_denom = np.maximum(data['high'] - data['low'], np.abs(data['open'] - data['close'].shift(1)))
    data['intraday_efficiency'] = (data['close'] - data['open']) / intraday_denom.replace(0, np.nan)
    
    # Gap Efficiency
    gap_eff_denom = np.abs(data['open'] - data['close'].shift(1))
    data['gap_efficiency'] = np.where(gap_eff_denom > 0, (data['close'] - data['open']) / gap_eff_denom, 0)
    
    # Price Response Asymmetry
    data['price_response_asymmetry'] = (data['high'] - data['close']) / (data['close'] - data['low']).replace(0, np.nan)
    
    # Multi-Scale Efficiency
    data['short_term_efficiency'] = (data['close'] - data['open']) / (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min())
    data['medium_term_efficiency'] = (data['close'] - data['open']) / (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())
    data['efficiency_ratio'] = data['short_term_efficiency'] / data['medium_term_efficiency'].replace(0, np.nan)
    
    # Efficiency-Volume Integration
    data['volume_weighted_efficiency'] = data['intraday_efficiency'] * (data['volume'] / data['volume'].shift(1))
    data['gap_volume_alignment'] = data['gap_efficiency'] * data['volume_response']
    data['asymmetric_efficiency'] = data['price_response_asymmetry'] * data['volume_persistence']
    
    # Fractal Divergence Framework
    # Volatility-Volume Divergence
    data['regime_divergence'] = np.sign(data['micro_vol'] - data['micro_vol'].shift(1)) * np.sign(data['volume_intensity'] - (data['volume'].shift(1) / data['volume'].shift(2)))
    data['persistence_divergence'] = data['persistence_signal'] * data['volume_momentum']
    data['efficiency_divergence'] = data['intraday_efficiency'] * data['volume_vol_ratio']
    
    # Multi-Scale Divergence
    data['short_term_divergence'] = data['volume_response'] * data['short_term_efficiency']
    data['medium_term_divergence'] = data['volume_persistence'] * data['medium_term_efficiency']
    data['cross_scale_divergence'] = data['efficiency_ratio'] * data['volume_fractal_ratio']
    
    # Fractal Break Signals
    data['volatility_break'] = np.abs(data['close'] - data['open']) / (data['high'].shift(1) - data['low'].shift(1))
    data['volume_break'] = (data['volume'] / data['volume'].shift(1)) * (data['volume'].shift(1) / data['volume'].shift(2))
    data['break_integration'] = data['volatility_break'] * data['volume_break'] * data['price_response_asymmetry']
    
    # Regime-Adaptive Synthesis
    # Regime-Specific Components
    data['high_vol_alpha'] = data['persistence_signal'] * data['volume_response'] * data['intraday_efficiency']
    data['low_vol_alpha'] = data['volume_momentum'] * data['gap_efficiency'] * data['medium_term_efficiency']
    data['transition_alpha'] = data['volume_vol_ratio'] * data['price_response_asymmetry'] * data['short_term_efficiency']
    
    # Multi-Scale Integration
    data['core_fractal'] = data['regime_divergence'] * data['amount_concentration']
    data['enhanced_core'] = data['core_fractal'] * data['break_integration']
    data['refined_core'] = data['enhanced_core'] * data['cross_scale_divergence']
    
    # Dynamic Regime Weighting
    data['regime_selection'] = np.where(
        data['high_vol_regime'], 
        data['high_vol_alpha'],
        np.where(
            data['low_vol_regime'],
            data['low_vol_alpha'],
            data['transition_alpha']
        )
    )
    
    data['volume_enhancement'] = data['regime_selection'] * data['volume_price_flow']
    data['fractal_confirmation'] = data['volume_enhancement'] * data['refined_core']
    
    # Final Alpha Output
    data['base_alpha'] = data['fractal_confirmation'] * data['asymmetric_efficiency']
    data['quality_adjustment'] = data['base_alpha'] * data['efficiency_ratio']
    data['final_alpha'] = data['quality_adjustment'] * data['volume_persistence']
    
    # Return the final alpha factor
    return data['final_alpha']
