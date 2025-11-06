import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Multi-Timeframe Acceleration Analysis
    # Short-term Acceleration (3-5 days)
    data['short_term_price_acceleration'] = (data['close'] / data['close'].shift(5) - 1) - (data['close'] / data['close'].shift(3) - 1)
    data['short_term_volume_acceleration'] = (data['volume'] / data['volume'].shift(5) - 1) - (data['volume'] / data['volume'].shift(3) - 1)
    
    # Medium-term Acceleration (5-12 days)
    data['medium_term_price_acceleration'] = (data['close'] / data['close'].shift(12) - 1) - (data['close'] / data['close'].shift(5) - 1)
    data['medium_term_volume_acceleration'] = (data['volume'] / data['volume'].shift(12) - 1) - (data['volume'] / data['volume'].shift(5) - 1)
    
    # Acceleration Divergence Detection
    data['price_acceleration_divergence'] = data['short_term_price_acceleration'] - data['medium_term_price_acceleration']
    data['volume_acceleration_divergence'] = data['short_term_volume_acceleration'] - data['medium_term_volume_acceleration']
    data['combined_acceleration_divergence'] = data['price_acceleration_divergence'] * data['volume_acceleration_divergence']
    
    # Volume-Flow Microstructure Integration
    # Flow-Based Acceleration Patterns
    data['flow'] = ((data['high'] + data['low'] + data['close']) / 3) * data['volume']
    data['short_term_flow_acceleration'] = (data['flow'] / data['flow'].shift(3) - 1) - (data['flow'] / data['flow'].shift(6) - 1)
    data['flow_volatility'] = (data['flow'] / data['flow'].shift(1) - 1).rolling(window=5, min_periods=3).std()
    
    # Volume-Amount Divergence Analysis
    data['volume_trend'] = data['volume'] / data['volume'].shift(3) - 1
    data['amount_trend'] = data['amount'] / data['amount'].shift(3) - 1
    data['volume_amount_divergence'] = (data['volume_trend'] - data['amount_trend']) * np.sign(data['volume_trend'])
    
    # Flow-Volume Confirmation
    data['flow_volume_alignment'] = np.sign(data['short_term_flow_acceleration']) == np.sign(data['volume_acceleration_divergence'])
    data['flow_volume_strength'] = np.abs(data['short_term_flow_acceleration']) * np.abs(data['volume_acceleration_divergence'])
    data['confirmation_score'] = data['flow_volume_alignment'].astype(int) * data['flow_volume_strength']
    
    # Volatility-Regime Context
    # Volatility-Based Regime Signals
    returns = data['close'] / data['close'].shift(1) - 1
    data['short_term_volatility'] = returns.rolling(window=5, min_periods=3).std()
    data['medium_term_volatility'] = returns.rolling(window=12, min_periods=5).std()
    data['volatility_regime'] = data['short_term_volatility'] / data['medium_term_volatility']
    
    # Range Efficiency Analysis
    data['price_change_5d'] = data['close'] / data['close'].shift(5) - 1
    data['daily_returns_abs_sum'] = returns.rolling(window=5, min_periods=3).apply(lambda x: np.sum(np.abs(x)), raw=True)
    data['range_efficiency'] = data['price_change_5d'] / data['daily_returns_abs_sum']
    
    # Efficiency Persistence
    data['range_efficiency_sign'] = np.sign(data['range_efficiency'])
    data['efficiency_persistence'] = data['range_efficiency_sign'].rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    
    # Volatility Adjustment
    data['volatility_multiplier'] = 1 + (data['volatility_regime'] - 1) * 0.5
    data['efficiency_weight'] = data['range_efficiency'] * data['efficiency_persistence']
    
    # Signal Integration and Enhancement
    # Core Divergence Component
    data['base_divergence_score'] = data['combined_acceleration_divergence'] * data['confirmation_score']
    
    # Regime-Adaptive Weighting
    data['volatility_weighted_divergence'] = data['base_divergence_score'] * data['volatility_multiplier']
    data['efficiency_weighted_confirmation'] = data['volatility_weighted_divergence'] * data['efficiency_weight']
    data['regime_optimized_signal'] = data['efficiency_weighted_confirmation'] * (1 + np.abs(data['volatility_regime'] - 1))
    
    # Intraday Validation
    # Intraday Strength Metrics
    data['intraday_return'] = data['close'] / data['open'] - 1
    data['intraday_range'] = (data['high'] - data['low']) / data['open']
    data['intraday_efficiency'] = np.abs(data['intraday_return']) / data['intraday_range']
    
    # Intraday Confirmation
    data['direction_confirmation'] = np.sign(data['regime_optimized_signal']) == np.sign(data['intraday_return'])
    data['strength_confirmation'] = data['intraday_efficiency'] > 0.5
    data['combined_intraday_score'] = data['direction_confirmation'].astype(int) * data['strength_confirmation'].astype(int)
    
    # Intraday Enhanced Signal
    data['intraday_enhanced_signal'] = data['regime_optimized_signal'] * (1 + data['combined_intraday_score'] * 0.3)
    
    # Final Alpha Factor Generation
    # Persistence Validation
    data['price_acceleration_sign'] = np.sign(data['short_term_price_acceleration'])
    data['acceleration_persistence'] = data['price_acceleration_sign'].rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    
    data['flow_acceleration_sign'] = np.sign(data['short_term_flow_acceleration'])
    data['flow_persistence'] = data['flow_acceleration_sign'].rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x == x.iloc[-1]) if len(x) > 0 else 0, raw=False
    )
    
    data['combined_persistence'] = (data['acceleration_persistence'] + data['flow_persistence']) / 10
    
    # Enhanced Alpha Factor
    data['enhanced_alpha_factor'] = data['intraday_enhanced_signal'] * (1 + data['combined_persistence'])
    
    return data['enhanced_alpha_factor']
