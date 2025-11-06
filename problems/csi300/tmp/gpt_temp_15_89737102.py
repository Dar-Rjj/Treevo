import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate basic components
    data['range'] = data['high'] - data['low']
    data['prev_close'] = data['close'].shift(1)
    data['return'] = data['close'] / data['prev_close'] - 1
    data['prev_return'] = data['return'].shift(1)
    data['prev_range'] = data['range'].shift(1)
    data['prev_volume'] = data['volume'].shift(1)
    data['prev_amount'] = data['amount'].shift(1)
    
    # Volatility-Liquidity Fractal Momentum
    # Multi-scale Fractal Analysis
    # Volatility Fractal Components
    data['range_5d_avg'] = data['range'].rolling(window=5).mean()
    data['intraday_vol_fractal'] = data['range'] / data['range_5d_avg'].shift(1)
    data['overnight_vol_fractal'] = abs(data['open'] - data['prev_close']) / data['range_5d_avg'].shift(1)
    data['closing_vol_fractal'] = abs(data['close'] - data['open']) / data['range']
    
    # Liquidity Fractal Components
    data['volume_5d_avg'] = data['volume'].rolling(window=5).mean()
    data['amount_5d_avg'] = data['amount'].rolling(window=5).mean()
    data['volume_fractal'] = data['volume'] / data['volume_5d_avg'].shift(1)
    data['amount_fractal'] = data['amount'] / data['amount_5d_avg'].shift(1)
    
    # Estimate volume concentration (using first hour approximation)
    data['volume_concentration'] = data['volume'].rolling(window=3).apply(
        lambda x: x.iloc[0] / x.sum() if x.sum() > 0 else 0
    )
    
    # Momentum Detection & Enhancement
    # Price Momentum Patterns
    data['morning_high'] = data['high'].rolling(window=3).apply(
        lambda x: x.iloc[0] if len(x) == 3 else np.nan
    )
    data['intraday_momentum'] = data['close'] / data['morning_high'] - 1
    data['overnight_gap_momentum'] = data['open'] / data['prev_close'] - 1
    data['reversal_strength'] = (data['close'] - data['open']) / (data['range'] + 1e-8)
    
    # Fractal Momentum Synthesis
    volatility_fractal_components = (data['intraday_vol_fractal'] + 
                                   data['overnight_vol_fractal'] + 
                                   data['closing_vol_fractal']) / 3
    liquidity_fractal_components = (data['volume_fractal'] + 
                                  data['amount_fractal']) / 2
    data['volatility_liquidity_phase'] = volatility_fractal_components * liquidity_fractal_components
    
    price_momentum_patterns = (data['intraday_momentum'] + 
                             data['overnight_gap_momentum'] + 
                             data['reversal_strength']) / 3
    data['momentum_amplification'] = price_momentum_patterns * data['volatility_liquidity_phase']
    data['fractal_momentum_score'] = data['momentum_amplification'] * data['volume_concentration']
    
    # Efficiency-Transition Discovery Factor
    # Market Efficiency Dynamics
    data['prev_day_range'] = data['range'].shift(1)
    data['opening_efficiency'] = abs(data['open'] - data['prev_close']) / (data['prev_day_range'] + 1e-8)
    data['closing_efficiency'] = abs(data['close'] - data['open']) / (data['range'] + 1e-8)
    data['intraday_efficiency_ratio'] = data['closing_efficiency'] / (data['opening_efficiency'] + 1e-8)
    
    # Transition Detection System
    data['volume_10d_avg'] = data['volume'].rolling(window=10).mean()
    data['range_10d_avg'] = data['range'].rolling(window=10).mean()
    data['volume_regime_break'] = data['volume'] / data['volume_10d_avg'].shift(1)
    data['volatility_regime_break'] = data['range'] / data['range_10d_avg'].shift(1)
    data['efficiency_transition'] = data['intraday_efficiency_ratio'] * data['volume_regime_break']
    
    # Discovery Momentum
    data['return_3d'] = data['close'] / data['close'].shift(3) - 1
    data['pre_transition_behavior'] = data['return_3d'].shift(3)
    data['post_transition_behavior'] = data['return_3d']
    data['discovery_score'] = (data['efficiency_transition'] * data['post_transition_behavior'] / 
                             (abs(data['pre_transition_behavior']) + 1e-8))
    
    # Range-Volume Asymmetric Persistence
    # Asymmetric Discovery Components
    data['range_discovery_asymmetry'] = data['range'] / data['prev_range'] - 1
    data['volume_discovery_asymmetry'] = data['volume'] / data['prev_volume'] - 1
    data['price_discovery_asymmetry'] = data['return'] / (abs(data['prev_return']) + 1e-8) - 1
    
    # Persistence Measurement
    data['range_volume_phase'] = data['range_discovery_asymmetry'] * data['volume_discovery_asymmetry']
    data['price_response'] = data['price_discovery_asymmetry'] * data['range_volume_phase']
    
    data['volume_5d_window'] = data['volume'].rolling(window=5).mean()
    data['volume_quality_confirmation'] = data['volume'] / data['volume_5d_window']
    
    data['morning_persistence'] = data['close'] / data['morning_high'] - 1
    
    opening_range_discovery = data['range_discovery_asymmetry'].rolling(window=3).apply(
        lambda x: x.iloc[0] if len(x) == 3 else np.nan
    )
    closing_range_discovery = data['range_discovery_asymmetry']
    data['intraday_persistence'] = closing_range_discovery / (abs(opening_range_discovery) + 1e-8)
    
    data['asymmetric_persistence_score'] = (data['price_response'] * 
                                          data['volume_quality_confirmation'] * 
                                          data['morning_persistence'])
    
    # Combine all factors with appropriate weights
    factor = (0.25 * data['fractal_momentum_score'] +
              0.25 * data['discovery_score'] +
              0.25 * data['asymmetric_persistence_score'])
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.fillna(method='ffill').fillna(0)
    
    return factor
