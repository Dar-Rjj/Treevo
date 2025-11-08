import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Price-Volume Divergence Regime Detection
    df = df.copy()
    
    # Calculate price and volume swings
    df['price_swing'] = (df['high'] - df['low']) / df['close'].shift(1)
    df['volume_swing'] = df['volume'] / df['volume'].shift(1) - 1
    
    # Replace infinite values and handle division by zero
    df['price_swing'] = df['price_swing'].replace([np.inf, -np.inf], np.nan)
    df['volume_swing'] = df['volume_swing'].replace([np.inf, -np.inf], np.nan)
    
    # Calculate divergence angle
    df['divergence_angle'] = np.arctan(df['volume_swing'] / df['price_swing'])
    
    # Identify divergence patterns
    df['price_lower_low'] = (df['low'] < df['low'].shift(1)) & (df['low'].shift(1) < df['low'].shift(2))
    df['volume_higher_low'] = (df['volume'] > df['volume'].shift(1)) & (df['volume'].shift(1) > df['volume'].shift(2))
    df['price_higher_high'] = (df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))
    df['volume_lower_high'] = (df['volume'] < df['volume'].shift(1)) & (df['volume'].shift(1) < df['volume'].shift(2))
    
    df['bullish_div'] = df['price_lower_low'] & df['volume_higher_low']
    df['bearish_div'] = df['price_higher_high'] & df['volume_lower_high']
    
    # Calculate consecutive divergence days
    df['bullish_consecutive'] = df['bullish_div'].groupby((~df['bullish_div']).cumsum()).cumcount()
    df['bearish_consecutive'] = df['bearish_div'].groupby((~df['bearish_div']).cumsum()).cumcount()
    
    # Divergence magnitude trend
    df['divergence_trend'] = df['divergence_angle'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / (np.std(x) + 1e-8))
    
    # Generate divergence factors
    df['bullish_factor'] = df['divergence_angle'] * df['bullish_consecutive']
    df['bearish_factor'] = -df['divergence_angle'] * df['volume_swing']
    
    # Intraday Momentum Fractality
    df['micro_momentum'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['meso_momentum'] = (df['close'] - df['close'].shift(1)) / (df['high'].shift(1) - df['low'].shift(1) + 1e-8)
    
    # Calculate rolling highs and lows for macro momentum
    df['max_high_5'] = df['high'].rolling(window=5).max()
    df['min_low_5'] = df['low'].rolling(window=5).min()
    df['macro_momentum'] = (df['close'] - df['close'].shift(5)) / (df['max_high_5'] - df['min_low_5'] + 1e-8)
    
    # Momentum alignment score
    df['alignment_score'] = np.sign(df['micro_momentum']) * np.sign(df['meso_momentum']) * np.sign(df['macro_momentum'])
    
    # Momentum gradient
    df['momentum_gradient'] = (df['macro_momentum'] - df['meso_momentum']) / (df['meso_momentum'] - df['micro_momentum'] + 1e-8)
    
    # Fractal dimension
    df['fractal_dimension'] = np.log(np.abs(df['macro_momentum']) + 1e-8) / np.log(np.abs(df['micro_momentum']) + 1e-8)
    
    # Coherent momentum periods
    df['momentum_coherent'] = (df['alignment_score'] > 0).astype(int)
    df['coherent_periods'] = df['momentum_coherent'].groupby((~df['momentum_coherent']).cumsum()).cumcount()
    
    # Generate momentum factors
    df['coherence_factor'] = df['alignment_score'] * df['fractal_dimension']
    df['gradient_factor'] = df['momentum_gradient'] * df['coherent_periods']
    
    # Liquidity Absorption Efficiency
    df['price_impact'] = (df['high'] - df['low']) / (df['volume'] + 1e-8)
    df['absorption_rate'] = df['amount'] / (df['high'] - df['low'] + 1e-8)
    df['price_return'] = (df['close'] - df['open']) / df['open']
    df['efficiency_ratio'] = df['price_return'] / (df['price_impact'] + 1e-8)
    
    # Absorption patterns
    df['efficiency_trend'] = df['efficiency_ratio'].rolling(window=3).apply(lambda x: (x[-1] - x[0]) / (np.std(x) + 1e-8))
    df['cumulative_efficiency'] = df['efficiency_ratio'].rolling(window=5).sum()
    
    # Generate efficiency factors
    df['core_efficiency'] = df['absorption_rate'] * df['efficiency_ratio']
    df['efficiency_momentum'] = df['efficiency_trend'] * df['cumulative_efficiency']
    
    # Volatility Compression Breakout Asymmetry
    df['avg_range_10'] = (df['high'] - df['low']).rolling(window=10).mean()
    df['avg_volume_10'] = df['volume'].rolling(window=10).mean()
    
    df['range_compression'] = (df['high'] - df['low']) / (df['avg_range_10'] + 1e-8)
    df['volume_compression'] = df['volume'] / (df['avg_volume_10'] + 1e-8)
    
    # Compression duration
    df['compressed'] = (df['range_compression'] < 0.8) & (df['volume_compression'] < 0.8)
    df['compression_duration'] = df['compressed'].groupby((~df['compressed']).cumsum()).cumcount()
    
    # Breakout probabilities
    df['upward_breakout_prob'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    df['downward_breakout_prob'] = (df['open'] - df['close']) / (df['high'] - df['low'] + 1e-8)
    df['asymmetry_ratio'] = df['upward_breakout_prob'] / (df['downward_breakout_prob'] + 1e-8)
    
    # Breakout quality
    df['volume_expansion'] = df['volume'] / df['volume'].shift(1)
    df['follow_through'] = (df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
    
    # Generate compression factors
    df['compression_factor'] = df['compression_duration'] * df['asymmetry_ratio']
    df['breakout_factor'] = df['volume_expansion'] * df['follow_through']
    
    # Momentum Regime Transition Signals
    df['price_momentum_change'] = (df['close'] / df['close'].shift(1) - 1) / (df['close'].shift(1) / df['close'].shift(2) - 1 + 1e-8) - 1
    df['volume_trend'] = df['volume'].rolling(window=5).apply(lambda x: (x[-1] - x[0]) / x[0])
    df['volume_momentum_change'] = df['volume_trend'] / (df['volume_trend'].shift(1) + 1e-8) - 1
    
    # Transition magnitude
    df['transition_magnitude'] = np.sqrt(df['price_momentum_change']**2 + df['volume_momentum_change']**2)
    
    # Regime persistence
    df['regime_stable'] = (df['price_momentum_change'].abs() < 0.1) & (df['volume_momentum_change'].abs() < 0.1)
    df['regime_persistence'] = df['regime_stable'].groupby((~df['regime_stable']).cumsum()).cumcount()
    
    # Transition smoothness
    df['transition_smoothness'] = 1 / (df['price_momentum_change'].rolling(window=3).std() + df['volume_momentum_change'].rolling(window=3).std() + 1e-8)
    
    # Generate transition factors
    df['early_detection'] = df['transition_magnitude'] * df['regime_persistence']
    df['quality_factor'] = df['transition_smoothness'] * df['transition_magnitude']
    
    # Combine all factors with weights
    final_factor = (
        0.25 * df['bullish_factor'] + 
        0.25 * df['bearish_factor'] +
        0.15 * df['coherence_factor'] +
        0.15 * df['gradient_factor'] +
        0.10 * df['core_efficiency'] +
        0.10 * df['efficiency_momentum'] +
        0.05 * df['compression_factor'] +
        0.05 * df['breakout_factor'] +
        0.05 * df['early_detection'] +
        0.05 * df['quality_factor']
    )
    
    return final_factor
