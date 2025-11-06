import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Fractal Momentum Reversal with Multi-Scale Confirmation
    """
    data = df.copy()
    
    # Helper function to calculate fractal levels
    def calculate_fractal_levels(window_data, method='median'):
        if method == 'median':
            return window_data.median()
        elif method == 'mean':
            return window_data.mean()
        else:
            return window_data.iloc[-1]  # Last value
    
    # Initialize fractal columns
    data['fractal_high_2'] = data['high'].rolling(window=2, min_periods=2).apply(
        lambda x: calculate_fractal_levels(x), raw=False)
    data['fractal_low_2'] = data['low'].rolling(window=2, min_periods=2).apply(
        lambda x: calculate_fractal_levels(x), raw=False)
    data['fractal_low_10'] = data['low'].rolling(window=10, min_periods=10).apply(
        lambda x: calculate_fractal_levels(x), raw=False)
    data['fractal_high_30'] = data['high'].rolling(window=30, min_periods=30).apply(
        lambda x: calculate_fractal_levels(x), raw=False)
    data['fractal_close_5'] = data['close'].rolling(window=5, min_periods=5).apply(
        lambda x: calculate_fractal_levels(x), raw=False)
    data['fractal_close_20'] = data['close'].rolling(window=20, min_periods=20).apply(
        lambda x: calculate_fractal_levels(x), raw=False)
    data['fractal_close_1'] = data['close'].shift(1)
    data['fractal_open'] = data['open'].rolling(window=2, min_periods=2).apply(
        lambda x: calculate_fractal_levels(x), raw=False)
    
    # Volume fractal moving average
    data['volume_fractal_ma'] = data['volume'].rolling(window=5, min_periods=5).mean()
    
    # Multi-Horizon Fractal Momentum Analysis
    data['short_term_momentum'] = ((data['close'] - data['fractal_low_2'].shift(2)) * 
                                  (data['high'] - data['low']))
    data['medium_term_momentum'] = ((data['close'] - data['fractal_low_10'].shift(10)) * 
                                   (data['high'] - data['low']))
    data['long_term_momentum'] = ((data['close'] - data['fractal_high_30'].shift(30)) * 
                                 (data['high'] - data['low']))
    
    # Fractal Momentum Divergence
    data['momentum_divergence'] = (
        np.sign(data['short_term_momentum']) + 
        np.sign(data['medium_term_momentum']) + 
        np.sign(data['long_term_momentum'])
    )
    
    # Intraday Extremes with Fractal Volatility Adjustment
    data['high_to_fractal_ratio'] = (data['high'] - data['fractal_high_30']) / data['close']
    data['low_to_fractal_ratio'] = (data['low'] - data['fractal_low_10']) / data['close']
    
    # Fractal Volatility Components
    data['fractal_returns'] = data['fractal_close_1'].pct_change()
    data['historical_fractal_vol'] = data['fractal_returns'].rolling(window=10, min_periods=10).std()
    data['intraday_fractal_vol'] = (data['high'] - data['low']) / data['fractal_close_1']
    
    # Fractal Volatility-Adjusted Extreme Signals
    fractal_vol_adj = np.where(data['historical_fractal_vol'] > 0, 
                              data['historical_fractal_vol'], 0.01)
    data['high_extreme_momentum'] = (data['high_to_fractal_ratio'] * data['volume'] / 
                                    fractal_vol_adj)
    data['low_extreme_momentum'] = (data['low_to_fractal_ratio'] * data['volume'] / 
                                   fractal_vol_adj)
    
    # Volume-Fractal Pressure Accumulation System
    data['volume_fractal_surprise'] = data['volume'] - data['volume_fractal_ma']
    data['volume_fractal_ratio'] = data['volume'] / data['volume_fractal_ma']
    
    data['fractal_reversal_pressure'] = (np.sign(data['close'] - data['fractal_open']) * 
                                        (data['high'] - data['low']))
    data['fractal_pressure_intensity'] = (np.abs(data['close'] - data['fractal_open']) / 
                                         (data['high'] - data['low']).replace(0, 0.001))
    
    # Volume-Fractal Weighted Pressure Accumulation
    data['weighted_pressure'] = (data['fractal_reversal_pressure'] * 
                                data['volume_fractal_ratio'] * 
                                data['fractal_pressure_intensity'])
    data['pressure_accumulation'] = data['weighted_pressure'].rolling(window=5, min_periods=5).sum()
    
    # Multi-Timeframe Fractal Reversal Efficiency
    data['short_fractal_momentum'] = data['close'].shift(1) - data['fractal_close_5'].shift(5)
    data['medium_fractal_momentum'] = data['close'].shift(1) - data['fractal_close_20'].shift(20)
    
    data['fractal_opening_efficiency'] = (np.abs(data['open'] - data['fractal_close_1']) / 
                                         (data['high'] - data['low']).replace(0, 0.001))
    data['intraday_fractal_reversal_efficiency'] = (np.sign(data['close'] - data['fractal_open']) * 
                                                   np.sign(data['high'] - data['low']))
    data['fractal_price_amplitude'] = (data['high'] - data['low']) / data['fractal_close_1']
    
    # Fractal Efficiency-Weighted Reversal Signals
    data['efficiency_weighted_signal'] = (
        data['intraday_fractal_reversal_efficiency'] * 
        data['fractal_opening_efficiency'] * 
        data['fractal_price_amplitude']
    )
    
    # Volatility-Fractal Weighted Reversal Score
    data['volatility_fractal_reversal'] = (
        (data['high_extreme_momentum'] + data['low_extreme_momentum']) *
        (data['short_term_momentum'] + data['medium_term_momentum'] + data['long_term_momentum']) *
        data['volume_fractal_ratio'] *
        data['pressure_accumulation'] *
        data['momentum_divergence']
    ) / (data['historical_fractal_vol'].replace(0, 0.01) + 0.001)
    
    # Fractal-Confirmed Alpha Signal
    data['fractal_coherence_multiplier'] = (
        np.abs(data['momentum_divergence']) *
        data['fractal_pressure_intensity'] *
        data['fractal_opening_efficiency']
    )
    
    # Final alpha factor
    alpha_factor = (
        data['volatility_fractal_reversal'] *
        data['fractal_coherence_multiplier'] *
        data['efficiency_weighted_signal']
    )
    
    return alpha_factor
