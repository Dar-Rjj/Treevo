import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily range
    data['range'] = data['high'] - data['low']
    
    # Multi-timeframe Elasticity Calculation
    data['avg_range_3d'] = data['range'].rolling(window=3, min_periods=3).mean()
    data['avg_range_10d'] = data['range'].rolling(window=10, min_periods=10).mean()
    
    data['elasticity'] = (data['range'] / data['avg_range_3d']) - 1
    data['elasticity_ratio'] = (data['avg_range_3d'] / data['avg_range_10d']) - 1
    
    # Elasticity Regime Classification
    conditions = [
        (data['elasticity'] > 1.5) | (data['elasticity_ratio'] > 0.3),
        (data['elasticity'] < -0.5) | (data['elasticity_ratio'] < -0.2)
    ]
    choices = [2, 0]  # 2: High, 0: Low, 1: Normal (default)
    data['elasticity_regime'] = np.select(conditions, choices, default=1)
    
    # Efficiency Analysis
    # 5-day efficiency ratio
    data['abs_change'] = abs(data['close'] - data['close'].shift(1))
    data['sum_abs_changes_5d'] = data['abs_change'].rolling(window=5, min_periods=5).sum()
    data['net_change_5d'] = abs(data['close'] - data['close'].shift(5))
    data['efficiency_5d'] = data['net_change_5d'] / data['sum_abs_changes_5d']
    data['efficiency_5d'] = data['efficiency_5d'].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Intraday efficiency ratio
    data['intraday_efficiency'] = abs(data['close'] - data['open']) / data['range']
    data['intraday_efficiency'] = data['intraday_efficiency'].replace([np.inf, -np.inf], np.nan).fillna(0)
    data['intraday_efficiency_10d'] = data['intraday_efficiency'].rolling(window=10, min_periods=10).mean()
    
    # Volume-Efficiency Relationship
    data['volume_momentum'] = data['volume'] / data['volume'].shift(3)
    data['volume_momentum'] = data['volume_momentum'].replace([np.inf, -np.inf], np.nan).fillna(1)
    
    data['volume_weighted_efficiency'] = (data['efficiency_5d'] + data['intraday_efficiency']) / 2 * data['volume_momentum']
    
    # Efficiency-Volume Divergence
    data['efficiency_trend'] = data['efficiency_5d'].rolling(window=5, min_periods=5).mean()
    data['volume_trend'] = data['volume'].rolling(window=5, min_periods=5).mean()
    data['efficiency_divergence'] = (data['efficiency_5d'] - data['efficiency_trend']) - (data['volume_momentum'] - 1)
    
    # VWAP Integration
    data['vwap'] = (data['high'] + data['low'] + data['close']) / 3 * data['volume']
    data['price_deviation'] = data['close'] - data['vwap']
    
    # Momentum-Elasticity Synthesis
    # Intraday Momentum Components
    data['range_momentum'] = (data['high'] - data['low']) / data['open']
    data['direction_momentum'] = (data['close'] - data['open']) / data['open']
    data['combined_momentum'] = data['range_momentum'] * data['direction_momentum']
    
    # Elasticity Adjustment
    data['elasticity_normalized_momentum'] = data['combined_momentum'] / (1 + abs(data['elasticity']))
    
    # Volume-Confirmed Momentum Enhancement
    data['volume_acceleration'] = data['volume'] / data['volume'].shift(3) - 1
    data['amount_ratio'] = data['amount'] / data['amount'].rolling(window=5, min_periods=5).mean()
    
    data['volume_confirmed_momentum'] = data['elasticity_normalized_momentum'] * data['volume_acceleration'] * data['amount_ratio']
    
    # Efficiency-Momentum Integration with Regime-based Weighting
    efficiency_weight = np.select(
        [data['elasticity_regime'] == 2, data['elasticity_regime'] == 0],
        [0.7, 0.3],
        default=0.5
    )
    momentum_weight = 1 - efficiency_weight
    
    data['integrated_score'] = (
        efficiency_weight * data['volume_weighted_efficiency'] + 
        momentum_weight * data['volume_confirmed_momentum']
    )
    
    # Dynamic Signal Adjustment
    # Extreme value filtering
    data['extreme_adjustment'] = 1 - abs(data['elasticity'])
    data['extreme_adjustment'] = np.clip(data['extreme_adjustment'], 0.1, 1.0)
    
    # Volume-amount confidence overlay
    volume_quantile = data['volume'].rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    amount_quantile = data['amount'].rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['volume_amount_confidence'] = (volume_quantile + amount_quantile) / 2
    
    # Persistence weighting
    regime_persistence = (data['elasticity_regime'] == data['elasticity_regime'].shift(1)).astype(int)
    data['regime_streak'] = regime_persistence.groupby((regime_persistence != regime_persistence.shift()).cumsum()).cumsum()
    data['persistence_multiplier'] = 1 + (data['regime_streak'] * 0.1)
    
    # Final Alpha Factor Construction
    data['final_alpha'] = (
        data['integrated_score'] * 
        data['extreme_adjustment'] * 
        data['volume_amount_confidence'] * 
        data['persistence_multiplier']
    )
    
    # Clean up and return
    result = data['final_alpha'].replace([np.inf, -np.inf], np.nan).fillna(0)
    return result
