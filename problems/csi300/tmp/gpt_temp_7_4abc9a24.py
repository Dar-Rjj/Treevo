import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Timeframe Volatility Assessment
    data['short_term_vol'] = (data['high'] - data['low']) / (data['high'].shift(5) - data['low'].shift(5))
    data['medium_term_vol'] = data['high'].rolling(window=20).apply(lambda x: x.max() - x.min()).rolling(window=20).std()
    data['volatility_ratio'] = data['short_term_vol'] / data['medium_term_vol']
    
    # Regime Classification
    conditions = [
        data['volatility_ratio'] > 1.2,
        data['volatility_ratio'] < 0.8
    ]
    choices = ['high', 'low']
    data['vol_regime'] = np.select(conditions, choices, default='normal')
    
    # Regime-Specific Price Dynamics
    data['short_term_ret'] = (data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['medium_term_ret'] = (data['close'] - data['close'].shift(5)) / data['close'].shift(5)
    data['reversal_divergence'] = data['short_term_ret'] / data['medium_term_ret']
    
    # Regime-Adaptive Price Component
    data['high_vol_price'] = (data['close'] - data['close'].shift(3)) / (data['high'].rolling(window=3).max() - data['low'].rolling(window=3).min()).rolling(window=3).std()
    data['low_vol_price'] = (data['close'] - data['close'].shift(15)) * (data['high'].rolling(window=20).max() - data['low'].rolling(window=20).min()).rolling(window=20).std()
    
    # Volume Dynamics Analysis
    data['volume_acceleration'] = (data['volume'].rolling(window=5).sum() / data['volume'].rolling(window=10).sum()) - 1
    data['volume_trend_ratio'] = data['volume'].rolling(window=5).sum() / data['volume'].rolling(window=20).sum()
    data['volume_momentum'] = (data['volume'] / data['volume'].shift(5)) - 1
    
    # Price-Volume Interaction
    data['price_volume_divergence'] = np.sign(data['close'] - data['close'].shift(1)) - np.sign(data['volume'] / data['volume'].rolling(window=20).sum())
    data['volume_driven_return'] = ((data['close'] - data['open']) / data['close'].shift(1)) / ((data['close'] - data['close'].shift(1)) / data['close'].shift(1))
    data['convergence_score'] = data['volume_driven_return'] * data['volume_trend_ratio']
    
    # Regime-Adaptive Price Component Final
    data['regime_adaptive_price'] = np.where(
        data['vol_regime'] == 'high', data['high_vol_price'],
        np.where(data['vol_regime'] == 'low', data['low_vol_price'],
                data['reversal_divergence'] * data['volume_acceleration'])
    )
    
    # Volume Confirmation Signal
    data['volume_confirmation'] = np.sign(data['regime_adaptive_price']) * data['volume_trend_ratio']
    data['divergence_amplifier'] = -(data['price_volume_divergence'] * data['volume_momentum'])
    
    # Intraday Market Structure
    data['opening_gap'] = (data['open'] / data['close'].shift(1)) - 1
    data['daily_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['gap_absorption'] = (data['close'] - data['open']) / (data['open'] - data['close'].shift(1))
    
    data['high_to_close_ratio'] = data['high'] / data['close']
    data['low_to_close_ratio'] = data['low'] / data['close']
    data['range_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    
    data['volume_to_volatility'] = data['volume'] / (data['high'] - data['low'])
    data['liquidity_trend'] = (data['volume_to_volatility'] / data['volume_to_volatility'].shift(1)) - 1
    data['range_consistency'] = (data['high'] - data['low']) / (data['high'].rolling(window=5).max() - data['low'].rolling(window=5).min())
    
    # Multi-Component Integration
    data['volatility_adjusted_price'] = data['regime_adaptive_price'] / (data['high'] - data['low'])
    data['volume_enhanced_component'] = data['volatility_adjusted_price'] * data['volume_confirmation']
    data['divergence_component'] = data['volume_enhanced_component'] * data['divergence_amplifier']
    
    data['high_component'] = data['high_to_close_ratio'] * data['divergence_component'] * data['liquidity_trend']
    data['low_component'] = data['low_to_close_ratio'] * data['divergence_component'] * data['liquidity_trend']
    data['session_signal'] = data['opening_gap'] * data['range_efficiency']
    
    # Regime-Adaptive Weighting
    def regime_weighted_component(row):
        if row['vol_regime'] == 'high':
            return row['convergence_score'] * 0.7 + row['divergence_component'] * 0.3
        elif row['vol_regime'] == 'low':
            return row['divergence_component'] * 0.7 + row['daily_strength'] * 0.3
        else:
            return (row['divergence_component'] + row['session_signal']) * 0.5
    
    data['regime_weighted_component'] = data.apply(regime_weighted_component, axis=1)
    
    # Final Alpha Signal Construction
    data['primary_signal'] = data['regime_weighted_component'] * data['volume_acceleration']
    data['validation_layer'] = data['primary_signal'] * data['volume_driven_return']
    data['stability_filter'] = data['validation_layer'] * (1 - data['range_consistency'])
    
    # Final signal
    alpha_signal = data['stability_filter']
    
    return alpha_signal
