import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Regime Gap-Pressure Momentum Factor
    Combines gap dynamics, volatility regime detection, multi-timeframe momentum,
    breakout patterns, volume-price dynamics, and microstructure efficiency
    """
    data = df.copy()
    
    # Gap Dynamics Analysis
    data['gap_magnitude'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    
    # Estimate bid/ask volume using amount and volume (approximation)
    avg_price = (data['high'] + data['low'] + data['close']) / 3
    data['implied_tick'] = np.where(data['close'] > data['open'], 1, -1)
    data['gap_pressure'] = data['implied_tick'] * data['volume'] / data['volume'].rolling(window=5, min_periods=1).mean()
    data['gap_pressure_alignment'] = data['gap_magnitude'] * data['gap_pressure']
    
    # Volatility Regime Detection
    data['prev_range'] = data['high'].shift(1) - data['low'].shift(1)
    data['curr_range'] = data['high'] - data['low']
    data['range_expansion'] = data['curr_range'] / data['prev_range']
    data['gap_volatility'] = abs(data['open'] - data['close'].shift(1)) / data['prev_range']
    
    # Regime Classification
    conditions = [
        data['range_expansion'] > 1.2,
        data['range_expansion'] < 0.8
    ]
    choices = ['high', 'low']
    data['volatility_regime'] = np.select(conditions, choices, default='normal')
    
    # Multi-Timeframe Momentum
    data['short_term_momentum'] = (data['close'] - data['close'].shift(3)) / data['close'].shift(3)
    data['medium_term_momentum'] = (data['close'] - data['close'].shift(8)) / data['close'].shift(8)
    data['momentum_divergence'] = abs(data['short_term_momentum'] - data['medium_term_momentum'])
    
    # Breakout Pattern Analysis
    data['up_breakout'] = (data['high'] > data['high'].shift(1)).astype(int) * data['gap_pressure_alignment']
    data['down_breakout'] = (data['low'] < data['low'].shift(1)).astype(int) * data['gap_pressure_alignment']
    
    # Breakout Persistence (5-day consecutive count)
    up_breakout_series = (data['high'] > data['high'].shift(1)).astype(int)
    down_breakout_series = (data['low'] < data['low'].shift(1)).astype(int)
    
    up_consecutive = up_breakout_series.rolling(window=5, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == 1 and all(x.iloc[max(0, i-4):i+1] == 1)]), 
        raw=False
    )
    down_consecutive = down_breakout_series.rolling(window=5, min_periods=1).apply(
        lambda x: len([i for i in range(1, len(x)) if x.iloc[i] == 1 and all(x.iloc[max(0, i-4):i+1] == 1)]), 
        raw=False
    )
    data['breakout_persistence'] = (up_consecutive - down_consecutive) * data['gap_pressure_alignment']
    
    # Volume-Price Dynamics
    data['volume_momentum'] = (data['volume'] - data['volume'].shift(3)) / data['volume'].shift(3)
    data['volume_price_coherence'] = np.sign(data['short_term_momentum']) * np.sign(data['volume_momentum'])
    data['volume_volatility_ratio'] = data['volume'] / data['curr_range']
    
    # Microstructure Efficiency
    data['range_efficiency'] = abs(data['close'] - data['open']) / data['curr_range']
    data['pressure_weighted_range_efficiency'] = data['range_efficiency'] * data['gap_pressure']
    data['intraday_position'] = (data['close'] - (data['high'] + data['low'])/2) / ((data['high'] - data['low'])/2)
    
    # Regime-Adaptive Components
    data['high_vol_component'] = data['gap_pressure_alignment'] * data['breakout_persistence'] * data['gap_volatility']
    data['low_vol_component'] = data['range_efficiency'] * data['volume_volatility_ratio'] * data['gap_pressure_alignment']
    data['normal_vol_component'] = data['momentum_divergence'] * data['volume_price_coherence'] * data['intraday_position']
    
    # Factor Construction
    data['momentum_component'] = data['momentum_divergence'] * abs(data['volume_price_coherence'])
    data['gap_pressure_component'] = data['gap_pressure_alignment'] * data['breakout_persistence']
    data['microstructure_component'] = data['pressure_weighted_range_efficiency'] * data['intraday_position']
    data['volume_validation'] = data['volume_volatility_ratio'] / data['volume_volatility_ratio'].shift(1)
    
    # Regime-Adaptive Factor
    regime_conditions = [
        data['volatility_regime'] == 'high',
        data['volatility_regime'] == 'low'
    ]
    regime_choices = [
        data['high_vol_component'] * data['volume_validation'],
        data['low_vol_component'] * data['volume_validation']
    ]
    data['regime_adaptive_factor'] = np.select(regime_conditions, regime_choices, 
                                              default=data['normal_vol_component'] * data['volume_validation'])
    
    # Final Alpha Factor
    alpha_factor = data['regime_adaptive_factor'] * data['microstructure_component']
    
    # Clean and return
    alpha_series = alpha_factor.replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    return alpha_series
