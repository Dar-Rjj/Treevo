import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Multi-scale Microstructure Momentum Divergence Factor
    Combines price efficiency, volume dynamics, and trade size analysis across multiple timeframes
    to detect momentum divergences with microstructure confirmation.
    """
    data = df.copy()
    
    # Multi-timeframe Microstructure Analysis
    # Intraday Price Efficiency Patterns
    data['opening_efficiency'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    data['midday_momentum'] = ((data['high'] + data['low']) / 2 - data['open']) - (data['close'] - (data['high'] + data['low']) / 2)
    data['closing_pressure'] = ((data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8)) - ((data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8))
    
    # Volume-Weighted Price Dynamics
    data['vwap_ratio'] = data['amount'] / (data['volume'] + 1e-8)
    data['volume_weighted_returns'] = (
        data['volume'] * (data['close'] - data['open']) + 
        data['volume'].shift(1) * (data['close'].shift(1) - data['open'].shift(1)) + 
        data['volume'].shift(2) * (data['close'].shift(2) - data['open'].shift(2))
    ) / (data['volume'] + data['volume'].shift(1) + data['volume'].shift(2) + 1e-8)
    
    data['large_trade_dominance'] = data['vwap_ratio'] / (data['vwap_ratio'].shift(1) + 1e-8)
    data['volume_clustering'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) * (data['volume'].shift(1) / (data['volume'].shift(2) + 1e-8))
    
    # Trade Size Distribution Analysis
    data['avg_trade_size_momentum'] = (data['vwap_ratio'] / (data['vwap_ratio'].shift(3) + 1e-8)) - 1
    data['trade_size_volatility'] = np.abs(data['vwap_ratio'] - data['vwap_ratio'].shift(1)) / (data['vwap_ratio'].shift(1) + 1e-8)
    data['size_price_correlation'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['vwap_ratio'] - data['vwap_ratio'].shift(1))
    
    # Multi-scale Momentum Divergence Detection
    # Short-term vs Medium-term Momentum
    data['price_momentum_divergence'] = ((data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)) - ((data['close'] - data['close'].shift(5)) / (data['close'].shift(5) + 1e-8))
    data['volume_momentum_divergence'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) - (data['volume'] / (data['volume'].shift(5) + 1e-8))
    data['efficiency_momentum_divergence'] = (np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)) - (np.abs(data['close'].shift(3) - data['open'].shift(3)) / (data['high'].shift(3) - data['low'].shift(3) + 1e-8))
    
    # Microstructure vs Price Momentum
    data['efficiency_price_divergence'] = data['opening_efficiency'] * np.sign(data['close'] - data['close'].shift(1))
    data['volume_price_divergence'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) * np.sign(data['close'] - data['close'].shift(1))
    data['trade_size_price_divergence'] = (data['vwap_ratio'] / (data['vwap_ratio'].shift(1) + 1e-8)) * np.sign(data['close'] - data['close'].shift(1))
    
    # Cross-timeframe Confirmation
    divergence_columns = ['price_momentum_divergence', 'volume_momentum_divergence', 'efficiency_momentum_divergence',
                         'efficiency_price_divergence', 'volume_price_divergence', 'trade_size_price_divergence']
    
    data['multi_scale_alignment'] = sum((data[col] > 0).astype(int) for col in divergence_columns)
    data['divergence_strength'] = data[divergence_columns].abs().mean(axis=1)
    
    # Calculate consistency pattern (consecutive days with same divergence direction)
    data['price_divergence_direction'] = np.sign(data['price_momentum_divergence'])
    data['consistency_pattern'] = data['price_divergence_direction'].groupby(data.index).transform(
        lambda x: x.rolling(window=3, min_periods=1).apply(lambda y: len(set(y)) == 1 if len(y) == 3 else 0)
    )
    
    # Market Regime Independent Processing
    # Absolute Divergence Signals
    data['raw_divergence_score'] = data[divergence_columns].sum(axis=1)
    
    # Calculate historical significance weights (using rolling volatility)
    weights = {}
    for col in divergence_columns:
        rolling_vol = data[col].rolling(window=20, min_periods=10).std()
        weights[col] = 1 / (rolling_vol + 1e-8)
    
    # Normalize weights
    weight_sum = sum(weights.values())
    for col in weights:
        weights[col] = weights[col] / (weight_sum + 1e-8)
    
    data['weighted_divergence'] = sum(data[col] * weights[col] for col in divergence_columns)
    data['directional_divergence'] = sum(np.sign(data[col]) * np.abs(data[col]) for col in divergence_columns)
    
    # Microstructure Confirmation
    data['volume_confirmation'] = data['weighted_divergence'] * (1 + data['volume'] / (data['volume'].shift(1) + 1e-8) - 1)
    data['efficiency_confirmation'] = data['weighted_divergence'] * (1 + np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8))
    data['trade_size_confirmation'] = data['weighted_divergence'] * (1 + data['vwap_ratio'] / (data['vwap_ratio'].shift(1) + 1e-8) - 1)
    
    # Signal Quality Assessment
    data['divergence_persistence'] = data['price_divergence_direction'].groupby(data.index).transform(
        lambda x: (x == x.shift(1)).rolling(window=5, min_periods=1).sum()
    )
    
    aligned_signals = sum((data[col] * data['price_divergence_direction'] > 0).astype(int) for col in divergence_columns)
    conflicting_signals = sum((data[col] * data['price_divergence_direction'] < 0).astype(int) for col in divergence_columns)
    data['signal_clarity'] = aligned_signals / (aligned_signals + conflicting_signals + 1e-8)
    
    microstructure_factors = ['volume_confirmation', 'efficiency_confirmation', 'trade_size_confirmation']
    data['microstructure_support'] = sum((data[factor] * data['weighted_divergence'] > 0).astype(int) for factor in microstructure_factors)
    
    # Adaptive Signal Integration
    # Dynamic Weighting Scheme
    data['volatility_scaling'] = 1 + np.abs(data['close'] - data['close'].shift(1)) / (data['close'].shift(1) + 1e-8)
    data['volume_adjusted_magnitude'] = (data['volume'] / (data['volume'].shift(5) + 1e-8)) ** 0.5
    
    # Cross-factor Validation
    data['price_efficiency_alignment'] = np.sign(data['close'] - data['open']) * np.sign(data['close'] - data['close'].shift(1))
    data['volume_efficiency_correlation'] = (data['volume'] / (data['volume'].shift(1) + 1e-8)) * (np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8))
    data['trade_size_efficiency_relationship'] = (data['vwap_ratio'] / (data['vwap_ratio'].shift(1) + 1e-8)) * (np.abs(data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8))
    
    # Signal Refinement
    data['noise_filtered_signal'] = data['weighted_divergence'].where(data['microstructure_support'] >= 2, 0)
    data['strength_adjusted_signal'] = data['noise_filtered_signal'] * (1 + data['divergence_persistence'] * 0.1)
    data['confidence_weighted_signal'] = data['strength_adjusted_signal'] * (1 + data['signal_clarity'] * 0.3)
    
    # Final Composite Output
    core_divergence_signal = data['confidence_weighted_signal']
    microstructure_confirmation = core_divergence_signal * (1 + data['microstructure_support'] * 0.2)
    timeframe_integration = microstructure_confirmation * (1 + data['multi_scale_alignment'] * 0.15)
    
    # Final factor output
    factor = timeframe_integration
    
    # Clean up and return
    factor = factor.replace([np.inf, -np.inf], np.nan)
    factor = factor.fillna(method='ffill').fillna(0)
    
    return factor
