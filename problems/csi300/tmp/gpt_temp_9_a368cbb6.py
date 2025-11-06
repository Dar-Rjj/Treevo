import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    data = df.copy()
    
    # Calculate True Range for ultra-short (2-day)
    data['TR_ultra'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    
    # Ultra-short efficiency (2-day)
    data['eff_ultra'] = np.abs(data['close'] - data['close'].shift(2)) / data['TR_ultra']
    
    # Calculate True Range for short-term (5-day)
    data['high_5d'] = data['high'].rolling(window=5, min_periods=5).max()
    data['low_5d'] = data['low'].rolling(window=5, min_periods=5).min()
    data['TR_short'] = np.maximum(
        data['high_5d'] - data['low_5d'],
        np.maximum(
            np.abs(data['high_5d'] - data['close'].shift(5)),
            np.abs(data['low_5d'] - data['close'].shift(5))
        )
    )
    
    # Short-term efficiency (5-day)
    data['eff_short'] = np.abs(data['close'] - data['close'].shift(5)) / data['TR_short']
    
    # Calculate True Range for medium-term (20-day)
    data['high_20d'] = data['high'].rolling(window=20, min_periods=20).max()
    data['low_20d'] = data['low'].rolling(window=20, min_periods=20).min()
    data['TR_medium'] = np.maximum(
        data['high_20d'] - data['low_20d'],
        np.maximum(
            np.abs(data['high_20d'] - data['close'].shift(20)),
            np.abs(data['low_20d'] - data['close'].shift(20))
        )
    )
    
    # Medium-term efficiency (20-day)
    data['eff_medium'] = np.abs(data['close'] - data['close'].shift(20)) / data['TR_medium']
    
    # Efficiency acceleration patterns
    conditions = [
        (data['eff_ultra'] > data['eff_short']) & (data['eff_short'] > data['eff_medium']),
        (data['eff_ultra'] < data['eff_short']) & (data['eff_short'] < data['eff_medium'])
    ]
    choices = [1, -1]
    data['efficiency_accel'] = np.select(conditions, choices, default=0)
    
    # Order Flow Imbalance (5-day)
    price_change_sign = np.sign(data['close'] - data['close'].shift(1))
    data['order_flow'] = (
        (data['amount'].rolling(window=5, min_periods=5).apply(
            lambda x: np.sum(x * price_change_sign[x.index[-5:]]), raw=False
        )) / 
        data['amount'].rolling(window=5, min_periods=5).apply(lambda x: np.sum(np.abs(x)), raw=False)
    )
    
    # Volume-Price Alignment (5-day)
    volume_weighted_return = (
        data['volume'].rolling(window=5, min_periods=5).apply(
            lambda x: np.sum(x * (data.loc[x.index, 'close'] - data.loc[x.index, 'close'].shift(1))), raw=False
        ) / 
        data['volume'].rolling(window=5, min_periods=5).sum()
    )
    price_return_5d = data['close'] / data['close'].shift(5) - 1
    data['volume_price_align'] = volume_weighted_return / price_return_5d
    
    # Volume-Price Regime Detection
    data['regime_strength'] = np.abs(data['order_flow']) * np.abs(data['volume_price_align'])
    conditions_regime = [
        (data['regime_strength'] > data['regime_strength'].rolling(window=20, min_periods=20).quantile(0.7)) & 
        (data['order_flow'] * np.sign(data['volume_price_align']) > 0),
        (data['regime_strength'] > data['regime_strength'].rolling(window=20, min_periods=20).quantile(0.7)) & 
        (data['order_flow'] * np.sign(data['volume_price_align']) < 0)
    ]
    choices_regime = [1, -1]  # 1: High Conviction, -1: Contrarian
    data['volume_regime'] = np.select(conditions_regime, choices_regime, default=0)
    
    # Intraday Position
    data['intraday_position'] = (data['close'] - data['low']) / (data['high'] - data['low'])
    
    # Range Consolidation
    high_5d_range = data['high'].rolling(window=5, min_periods=5).max() - data['low'].rolling(window=5, min_periods=5).min()
    high_10d_range = data['high'].rolling(window=10, min_periods=10).max() - data['low'].rolling(window=10, min_periods=10).min()
    data['range_consolidation'] = 1 - (high_5d_range / high_10d_range)
    
    # Key Level Proximity
    data['resistance_proximity'] = (data['high_20d'] - data['close']) / data['close']
    data['support_proximity'] = (data['close'] - data['low_20d']) / data['close']
    data['key_level_adjust'] = np.minimum(data['resistance_proximity'], data['support_proximity'])
    
    # Synthesize Adaptive Alpha Factor
    # Base factor combining efficiency acceleration and volume-price alignment
    base_factor = data['efficiency_accel'] * data['volume_price_align'] * data['order_flow']
    
    # Apply dynamics scaling
    position_alignment = 2 * data['intraday_position'] - 1  # Convert to -1 to 1 range
    dynamics_factor = base_factor * position_alignment * data['range_consolidation'] * (1 + data['key_level_adjust'])
    
    # Final alpha factor with regime strength weighting
    alpha_factor = dynamics_factor * (1 + 0.5 * np.abs(data['volume_regime']))
    
    return alpha_factor
