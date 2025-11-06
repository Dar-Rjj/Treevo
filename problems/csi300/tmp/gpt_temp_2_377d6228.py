import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Fractal Momentum Structure
    # Short-Term Momentum (1-3 days)
    data['short_price_momentum'] = (data['close'] - data['close'].shift(2)) / data['close'].shift(2)
    data['short_volume_momentum'] = (data['volume'] - data['volume'].shift(2)) / np.maximum(data['volume'].shift(2), 1)
    data['short_amount_momentum'] = (data['amount'] - data['amount'].shift(2)) / np.maximum(data['amount'].shift(2), 1)
    
    # Medium-Term Momentum (5-10 days)
    data['price_ret'] = data['close'] / data['close'].shift(1) - 1
    data['volume_ret'] = data['volume'] / np.maximum(data['volume'].shift(1), 1) - 1
    data['amount_ret'] = data['amount'] / np.maximum(data['amount'].shift(1), 1) - 1
    
    data['medium_price_momentum'] = data['price_ret'].rolling(window=5, min_periods=3).sum()
    data['medium_volume_momentum'] = data['volume_ret'].rolling(window=5, min_periods=3).sum()
    data['medium_amount_momentum'] = data['amount_ret'].rolling(window=5, min_periods=3).sum()
    
    # Long-Term Momentum (20-30 days)
    data['long_price_momentum'] = data['close'] / data['close'].shift(20) - 1
    data['long_volume_momentum'] = data['volume'] / np.maximum(data['volume'].shift(20), 1) - 1
    data['long_amount_momentum'] = data['amount'] / np.maximum(data['amount'].shift(20), 1) - 1
    
    # Multi-Timeframe Divergence Detection
    # Short vs Medium Divergence
    data['price_divergence_sm'] = data['short_price_momentum'] - data['medium_price_momentum']
    data['volume_divergence_sm'] = data['short_volume_momentum'] - data['medium_volume_momentum']
    data['amount_divergence_sm'] = data['short_amount_momentum'] - data['medium_amount_momentum']
    
    # Medium vs Long Divergence
    data['price_divergence_ml'] = data['medium_price_momentum'] - data['long_price_momentum']
    data['volume_divergence_ml'] = data['medium_volume_momentum'] - data['long_volume_momentum']
    data['amount_divergence_ml'] = data['medium_amount_momentum'] - data['long_amount_momentum']
    
    # Cross-Asset Divergence Patterns
    data['price_volume_div_corr'] = data['price_divergence_sm'] * data['volume_divergence_sm']
    data['price_amount_div_consistency'] = data['price_divergence_sm'] * data['amount_divergence_sm']
    data['volume_amount_div_alignment'] = data['volume_divergence_sm'] * data['amount_divergence_sm']
    
    # Bidirectional Flow Imbalance Analysis
    # Upward Flow Strength
    data['price_up'] = (data['close'] > data['close'].shift(1)).astype(int)
    data['up_pressure'] = data['price_up'].rolling(window=5, min_periods=3).sum()
    
    # Create masks for conditional sums
    up_mask = data['price_up'] == 1
    down_mask = data['price_up'] == 0
    
    # Calculate conditional sums using rolling apply
    data['up_volume'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x[data['price_up'].iloc[len(data)-len(x):len(data)].values == 1]) if len(x) == 5 else np.nan
    )
    data['up_amount'] = data['amount'].rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x[data['price_up'].iloc[len(data)-len(x):len(data)].values == 1]) if len(x) == 5 else np.nan
    )
    
    # Downward Flow Resistance
    data['down_pressure'] = 5 - data['up_pressure']
    data['down_volume'] = data['volume'].rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x[data['price_up'].iloc[len(data)-len(x):len(data)].values == 0]) if len(x) == 5 else np.nan
    )
    data['down_amount'] = data['amount'].rolling(window=5, min_periods=3).apply(
        lambda x: np.sum(x[data['price_up'].iloc[len(data)-len(x):len(data)].values == 0]) if len(x) == 5 else np.nan
    )
    
    # Flow Imbalance Metrics
    data['net_price_pressure'] = data['up_pressure'] - data['down_pressure']
    data['volume_flow_ratio'] = data['up_volume'] / (data['up_volume'] + data['down_volume'] + 1e-8)
    data['amount_flow_dominance'] = (data['up_amount'] - data['down_amount']) / (data['up_amount'] + data['down_amount'] + 1e-8)
    
    # Divergence-Flow Convergence Framework
    # Momentum Divergence Confirmation
    data['sm_divergence_flow_alignment'] = data['price_divergence_sm'] * data['net_price_pressure']
    data['ml_divergence_flow_confirmation'] = data['price_divergence_ml'] * data['volume_flow_ratio']
    data['cross_asset_divergence_flow'] = (data['price_volume_div_corr'] + data['price_amount_div_consistency']) * data['amount_flow_dominance']
    
    # Flow Imbalance Reinforcement
    data['positive_divergence_upflow'] = np.where(data['price_divergence_sm'] > 0, data['price_divergence_sm'] * data['up_pressure'], 0)
    data['negative_divergence_downflow'] = np.where(data['price_divergence_sm'] < 0, data['price_divergence_sm'] * data['down_pressure'], 0)
    
    # Convergence Strength Measurement
    data['divergence_magnitude_flow'] = (abs(data['price_divergence_sm']) + abs(data['price_divergence_ml'])) * data['net_price_pressure']
    data['divergence_persistence_flow'] = (data['price_divergence_sm'].rolling(window=3, min_periods=2).std() + 1e-8) * data['volume_flow_ratio']
    data['cross_timeframe_convergence'] = (data['price_divergence_sm'] * data['price_divergence_ml']) * data['amount_flow_dominance']
    
    # Composite Factor Construction
    # Fractal Divergence Score
    divergence_weights = [0.4, 0.3, 0.3]  # Short-Medium, Medium-Long, Cross-asset
    data['fractal_divergence_score'] = (
        divergence_weights[0] * data['price_divergence_sm'] +
        divergence_weights[1] * data['price_divergence_ml'] +
        divergence_weights[2] * (data['price_volume_div_corr'] + data['volume_amount_div_alignment']) / 2
    )
    
    # Flow Imbalance Score
    flow_weights = [0.5, 0.3, 0.2]  # Net pressure, Volume ratio, Amount dominance
    data['flow_imbalance_score'] = (
        flow_weights[0] * data['net_price_pressure'] +
        flow_weights[1] * data['volume_flow_ratio'] +
        flow_weights[2] * data['amount_flow_dominance']
    )
    
    # Final Convergence Factor
    data['convergence_factor'] = (
        data['fractal_divergence_score'] * data['flow_imbalance_score'] *
        (1 + 0.1 * data['cross_timeframe_convergence']) *
        (1 + 0.05 * data['divergence_magnitude_flow'])
    )
    
    # Clean up intermediate columns
    result = data['convergence_factor'].copy()
    
    return result
