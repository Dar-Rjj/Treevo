import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Initialize all required columns
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Efficiency Metrics
    data['price_efficiency'] = (data['close'] - data['close'].shift(3)) / (
        abs(data['close'] - data['close'].shift(1)) + 
        abs(data['close'].shift(1) - data['close'].shift(2)) + 
        abs(data['close'].shift(2) - data['close'].shift(3))
    )
    
    data['range_efficiency'] = (data['close'] - data['close'].shift(3)) / (
        (data['high'] - data['low']) + 
        (data['high'].shift(1) - data['low'].shift(1)) + 
        (data['high'].shift(2) - data['low'].shift(2))
    )
    
    data['volatility_adjusted_efficiency'] = (data['close'] - data['close'].shift(1)) / (
        data['true_range'].rolling(window=5, min_periods=1).mean()
    )
    
    data['microstructure_efficiency'] = abs(data['close'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'])
    
    # Volatility Context
    data['volatility_regime'] = data['true_range'] / data['true_range'].rolling(window=5, min_periods=1).mean()
    
    data['volume_volatility_elasticity'] = (
        (data['volume'] / data['volume'].shift(1)) / 
        (data['true_range'] / data['true_range'].shift(1))
    )
    
    # Multi-Timeframe Acceleration
    data['price_acceleration'] = (
        (data['close'] / data['close'].shift(6) - 1) - 
        (data['close'] / data['close'].shift(3) - 1)
    )
    
    data['volume_acceleration'] = (
        (data['volume'] / data['volume'].shift(6) - 1) - 
        (data['volume'] / data['volume'].shift(3) - 1)
    )
    
    data['efficiency_acceleration'] = data['price_efficiency'] - data['price_efficiency'].shift(3)
    
    # Flow Momentum & Microstructure Integration
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    data['money_flow'] = data['typical_price'] * data['volume'] * np.sign(data['close'] - data['close'].shift(1))
    
    data['short_flow'] = data['money_flow'] - data['money_flow'].shift(3)
    data['medium_flow'] = data['money_flow'] - data['money_flow'].shift(6)
    data['flow_acceleration'] = data['short_flow'] - data['medium_flow']
    
    # Microstructure Anchoring
    data['trading_friction'] = abs(data['close'] - data['open']) / (data['amount'] / data['volume'])
    data['position_strength'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    data['opening_efficiency'] = abs(data['open'] - (data['high'] + data['low']) / 2) / (data['high'] - data['low'])
    
    # Efficiency-Weighted Momentum
    data['volatility_adaptive_momentum'] = (data['close'] - data['close'].shift(3)) / (
        data['true_range'].rolling(window=3, min_periods=1).mean()
    )
    
    data['flow_enhanced_momentum'] = data['volatility_adaptive_momentum'] * data['short_flow']
    data['efficiency_confirmed_momentum'] = data['flow_enhanced_momentum'] * data['price_efficiency']
    
    # Divergence Analysis
    data['efficiency_momentum_divergence'] = data['price_efficiency'] * data['volatility_adaptive_momentum']
    data['volume_efficiency_divergence'] = data['volume_acceleration'] * data['efficiency_acceleration']
    data['flow_volatility_divergence'] = data['flow_acceleration'] * data['volatility_regime']
    
    # Microstructure Confirmation
    data['position_efficiency_alignment'] = data['position_strength'] * data['range_efficiency']
    data['opening_closing_divergence'] = data['opening_efficiency'] - data['microstructure_efficiency']
    data['friction_efficiency_relationship'] = data['trading_friction'] * data['volatility_adjusted_efficiency']
    
    # Regime Transition Signals
    data['volatility_breakout'] = ((data['volatility_regime'] > 1.2) & (data['efficiency_acceleration'] > 0)).astype(float)
    data['flow_regime_change'] = data['flow_acceleration'] * data['volume_volatility_elasticity']
    data['microstructure_regime'] = ((data['trading_friction'] > data['trading_friction'].rolling(window=10).mean()) & 
                                   (data['microstructure_efficiency'] < data['microstructure_efficiency'].rolling(window=10).mean())).astype(float)
    
    # Liquidity Metrics
    data['amount_ratio'] = data['amount'] / data['amount'].rolling(window=3, min_periods=1).mean()
    
    # Volume persistence (count consecutive days with volume > 3-day average)
    volume_avg_3d = data['volume'].rolling(window=3, min_periods=1).mean()
    data['volume_persistence'] = 0
    for i in range(1, len(data)):
        if data['volume'].iloc[i] > volume_avg_3d.iloc[i]:
            data.loc[data.index[i], 'volume_persistence'] = data['volume_persistence'].iloc[i-1] + 1
    
    data['volume_concentration'] = data['volume'] / data['volume'].rolling(window=4, min_periods=1).mean()
    
    # Intraday Strength
    data['range_utilization'] = (data['close'] - data['open']) / (data['high'] - data['low'])
    data['close_position_strength'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Intraday consistency (count of range_utilization > 0.5 in last 3 days)
    data['intraday_consistency'] = (data['range_utilization'] > 0.5).rolling(window=3, min_periods=1).sum()
    
    # Combined Assessment
    data['liquidity_score'] = data['amount_ratio'] * data['volume_persistence']
    data['strength_score'] = data['position_strength'] * data['range_utilization']
    data['microstructure_quality'] = 1 - (data['trading_friction'] + data['opening_efficiency'])
    
    # Adaptive Alpha Generation
    # High Efficiency Flow Strategy
    high_efficiency_condition = (data['price_efficiency'] > 0.6) & (data['flow_acceleration'] > 0)
    high_efficiency_confirmation = (data['volume_efficiency_divergence'] > 0) & (data['strength_score'] > 0.3)
    high_efficiency_factor = data['efficiency_confirmed_momentum'] * high_efficiency_confirmation.astype(float) * data['liquidity_score']
    
    # Volatility-Regime Adaptive Strategy
    volatility_condition = (data['volatility_regime'] > 1.0) & (data['efficiency_acceleration'] > 0)
    volatility_confirmation = (data['microstructure_quality'] > 0.5) & (data['volume_concentration'] > 1.1)
    volatility_factor = (data['efficiency_momentum_divergence'] * data['flow_volatility_divergence']) * volatility_confirmation.astype(float) * data['volatility_regime']
    
    # Microstructure-Enhanced Divergence Strategy
    microstructure_condition = (data['opening_closing_divergence'] > 0) & (data['position_efficiency_alignment'] > 0)
    microstructure_confirmation = (data['trading_friction'] < 0.1) & (data['intraday_consistency'] > 1)
    microstructure_factor = (data['efficiency_momentum_divergence'] * data['flow_acceleration']) * microstructure_confirmation.astype(float) * data['microstructure_quality']
    
    # Combine strategies with conditions
    final_factor = np.zeros(len(data))
    
    # Apply each strategy only when its condition is met
    final_factor[high_efficiency_condition] = high_efficiency_factor[high_efficiency_condition]
    final_factor[volatility_condition & ~high_efficiency_condition] = volatility_factor[volatility_condition & ~high_efficiency_condition]
    final_factor[microstructure_condition & ~high_efficiency_condition & ~volatility_condition] = microstructure_factor[microstructure_condition & ~high_efficiency_condition & ~volatility_condition]
    
    # For days where no condition is met, use weighted average
    no_condition_mask = ~(high_efficiency_condition | volatility_condition | microstructure_condition)
    if no_condition_mask.any():
        weighted_avg = (
            data['efficiency_confirmed_momentum'] * 0.4 +
            data['efficiency_momentum_divergence'] * 0.3 +
            data['flow_enhanced_momentum'] * 0.3
        )
        final_factor[no_condition_mask] = weighted_avg[no_condition_mask]
    
    return pd.Series(final_factor, index=data.index)
