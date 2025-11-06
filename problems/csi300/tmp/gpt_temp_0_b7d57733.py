import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Intraday Momentum Components
    # Short-term Price Momentum
    data['momentum_5d'] = data['close'].shift(1) / data['close'].shift(5) - 1
    data['momentum_2d'] = data['close'].shift(1) / data['close'].shift(2) - 1
    
    # Intraday Strength
    data['intraday_strength'] = (data['close'].shift(1) - data['open'].shift(1)) / (data['high'].shift(1) - data['low'].shift(1))
    
    # Volume Efficiency Analysis
    # Price Movement Efficiency
    data['abs_return'] = abs(data['close'].shift(1) / data['close'].shift(2) - 1)
    data['true_range'] = np.maximum(
        data['high'].shift(1) - data['low'].shift(1),
        np.maximum(
            abs(data['high'].shift(1) - data['close'].shift(2)),
            abs(data['low'].shift(1) - data['close'].shift(2))
        )
    )
    data['efficiency'] = data['abs_return'] / data['true_range']
    
    # Volume-Adjusted Efficiency
    data['volume_adj_efficiency'] = data['efficiency'] * data['volume'].shift(1)
    data['eff_5d_avg'] = data['volume_adj_efficiency'].rolling(window=5, min_periods=1).mean().shift(1)
    data['efficiency_deviation'] = data['volume_adj_efficiency'] / data['eff_5d_avg'] - 1
    
    # Momentum Divergence Detection
    data['intraday_sign'] = np.sign(data['intraday_strength'])
    data['momentum_5d_sign'] = np.sign(data['momentum_5d'])
    
    # Momentum Acceleration
    momentum_denom = np.where(abs(data['momentum_5d']) > 1e-8, abs(data['momentum_5d']), 1e-8)
    data['momentum_acceleration'] = (data['momentum_2d'] - data['momentum_5d']) / momentum_denom
    
    # Volume-Volatility Confirmation
    # Volume Pattern Analysis
    data['volume_5d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean().shift(1)
    data['volume_10d_avg'] = data['volume'].rolling(window=5, min_periods=1).mean().shift(6)
    data['volume_ratio'] = data['volume_5d_avg'] / data['volume_10d_avg']
    
    # Volume Persistence
    data['volume_change'] = data['volume'].pct_change()
    data['volume_persistence'] = data['volume_change'].rolling(window=3, min_periods=1).apply(
        lambda x: np.sum(np.sign(x) == np.sign(x.iloc[0])) if len(x) > 0 else 0
    ).shift(1)
    
    # Volatility Regime
    data['vol_5d_avg'] = data['true_range'].rolling(window=5, min_periods=1).mean().shift(1)
    data['volatility_regime'] = np.where(data['true_range'] > data['vol_5d_avg'], 1, 0)
    
    # Adaptive Signal Combination
    # Volatility-Regime Weighted Momentum
    data['high_vol_weight'] = data['volatility_regime'] * (0.6 * data['intraday_strength'] + 0.4 * data['momentum_2d'])
    data['low_vol_weight'] = (1 - data['volatility_regime']) * (0.6 * data['momentum_5d'] + 0.4 * data['volume_persistence'])
    data['regime_weighted_momentum'] = data['high_vol_weight'] + data['low_vol_weight']
    
    # Volume Efficiency Confirmation
    efficiency_trend = np.sign(data['efficiency_deviation'].rolling(window=3, min_periods=1).mean().shift(1))
    volume_confirmation = np.sign(data['volume_ratio'] - 1) * efficiency_trend
    data['momentum_efficiency_composite'] = data['momentum_acceleration'] * data['efficiency_deviation'] * volume_confirmation
    
    # Generate Composite Alpha
    # Combine components with appropriate weights
    data['alpha_component1'] = data['regime_weighted_momentum'] * (1 + data['volume_persistence'] / 10)
    data['alpha_component2'] = data['momentum_efficiency_composite'] * (1 + abs(data['intraday_strength']))
    
    # Final alpha signal
    data['alpha'] = 0.5 * data['alpha_component1'] + 0.5 * data['alpha_component2']
    
    # Apply direction based on intraday strength and short-term momentum
    direction_signal = np.sign(data['intraday_sign'] + data['momentum_5d_sign'])
    data['final_alpha'] = data['alpha'] * direction_signal
    
    return data['final_alpha']
