import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Volume-Efficiency Momentum Divergence Factor
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Multi-Scale Volatility & Efficiency State Analysis
    # Volatility Structure Assessment
    data['daily_range'] = (data['high'] - data['low']) / data['close']
    
    # Short-term volatility (geometric mean of daily range from t-1 to t)
    data['vol_short'] = data['daily_range'].rolling(window=2).apply(
        lambda x: np.exp(np.mean(np.log(x))) if all(x > 0) else np.nan
    )
    
    # Medium-term volatility (geometric mean of daily range from t-5 to t)
    data['vol_medium'] = data['daily_range'].rolling(window=6).apply(
        lambda x: np.exp(np.mean(np.log(x))) if all(x > 0) else np.nan
    )
    
    # Volatility ratio for regime classification
    data['vol_ratio'] = data['vol_short'] / data['vol_medium']
    
    # Efficiency State Classification
    data['volume_efficiency'] = data['amount'] / (data['volume'] * data['close'])
    data['efficiency_momentum'] = data['volume_efficiency'] - data['volume_efficiency'].shift(5)
    
    # Efficiency consistency (sign agreement between daily efficiency changes)
    data['efficiency_change'] = data['volume_efficiency'].diff()
    data['efficiency_consistency'] = data['efficiency_change'].rolling(window=3).apply(
        lambda x: np.mean(np.sign(x) == np.sign(x.iloc[-1])) if len(x) == 3 else np.nan
    )
    
    # Regime Detection System
    conditions = [
        data['vol_ratio'] > 1.3,
        data['vol_ratio'] < 0.8,
        (data['vol_ratio'] >= 0.8) & (data['vol_ratio'] <= 1.3)
    ]
    choices = ['high_vol', 'low_vol', 'transition']
    data['vol_regime'] = np.select(conditions, choices, default='transition')
    
    # Volume-Efficiency Divergence Framework
    # Directional Volume Analysis
    data['return'] = data['close'].pct_change()
    data['up_volume'] = np.where(data['return'] > 0, data['volume'], 0)
    data['down_volume'] = np.where(data['return'] < 0, data['volume'], 0)
    
    data['up_volume_pressure'] = data['up_volume'].rolling(window=5).sum()
    data['down_volume_pressure'] = data['down_volume'].rolling(window=5).sum()
    data['volume_imbalance'] = data['up_volume_pressure'] / (data['down_volume_pressure'] + 1e-8)
    
    # Efficiency-Momentum Strength Assessment
    data['price_range_efficiency'] = ((data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)).rolling(window=3).mean()
    data['volume_adjusted_momentum'] = data['return'] / (data['volume'] + 1e-8)
    data['efficiency_weighted_momentum'] = data['return'] * data['volume_efficiency']
    
    # Divergence Detection
    data['price_efficiency_corr'] = data['return'].rolling(window=5).corr(data['volume_efficiency'])
    
    # Momentum-efficiency alignment
    data['momentum_direction'] = np.sign(data['return'])
    data['efficiency_trend'] = np.sign(data['efficiency_momentum'])
    data['momentum_efficiency_alignment'] = data['momentum_direction'] * data['efficiency_trend']
    
    # Divergence strength
    data['divergence_strength'] = np.abs(data['price_efficiency_corr']) * np.abs(data['efficiency_momentum'])
    
    # Regime-Adaptive Signal Processing
    # High Volatility Signal Enhancement
    high_vol_mask = data['vol_regime'] == 'high_vol'
    data['high_vol_signal'] = np.where(
        high_vol_mask,
        data['divergence_strength'] * data['efficiency_consistency'],
        0
    )
    
    # Low Volatility Signal Refinement
    low_vol_mask = data['vol_regime'] == 'low_vol'
    data['efficiency_trend_persistence'] = data['efficiency_trend'].rolling(window=5).mean()
    data['low_vol_signal'] = np.where(
        low_vol_mask,
        data['efficiency_trend_persistence'] * data['efficiency_consistency'],
        0
    )
    
    # Transition Regime Signal Balancing
    trans_mask = data['vol_regime'] == 'transition'
    data['efficiency_acceleration'] = data['efficiency_momentum'].diff()
    data['transition_signal'] = np.where(
        trans_mask,
        (data['divergence_strength'] + data['efficiency_trend_persistence']) / 2 * np.sign(data['efficiency_acceleration']),
        0
    )
    
    # Adaptive Factor Construction
    # Multi-Timeframe Signal Integration
    data['short_term_signal'] = (
        data['volume_adjusted_momentum'].rolling(window=2).mean() * 
        data['efficiency_consistency']
    )
    
    data['medium_term_signal'] = (
        data['efficiency_weighted_momentum'].rolling(window=6).mean() * 
        data['efficiency_trend_persistence']
    )
    
    # Regime-weighted combination
    regime_weights = {
        'high_vol': 0.6,
        'low_vol': 0.3,
        'transition': 0.5
    }
    
    data['regime_weight'] = data['vol_regime'].map(regime_weights)
    data['combined_signal'] = (
        data['regime_weight'] * data['short_term_signal'] + 
        (1 - data['regime_weight']) * data['medium_term_signal']
    )
    
    # Divergence Pattern Detection
    # Price-Volume-Efficiency Divergence
    data['positive_divergence'] = (
        (data['return'] > 0) & 
        (data['volume'] < data['volume'].shift(1)) & 
        (data['efficiency_momentum'] > 0)
    ).astype(int)
    
    data['negative_divergence'] = (
        (data['return'] < 0) & 
        (data['volume'] > data['volume'].shift(1)) & 
        (data['efficiency_momentum'] < 0)
    ).astype(int)
    
    data['confirmed_move'] = (
        (np.sign(data['return']) == np.sign(data['volume'].diff())) & 
        (np.sign(data['return']) == np.sign(data['efficiency_momentum']))
    ).astype(int)
    
    # Acceleration-Sustainability Divergence
    data['price_acceleration'] = data['return'].diff()
    data['efficiency_sustainability'] = data['efficiency_momentum'].rolling(window=3).std()
    
    # Composite Factor Generation
    # Volatility-Normalized Efficiency Divergence Score
    volatility_normalizers = {
        'high_vol': 2.0,
        'low_vol': 0.5,
        'transition': 1.0
    }
    data['vol_normalizer'] = data['vol_regime'].map(volatility_normalizers)
    
    data['divergence_score'] = (
        data['combined_signal'] * data['vol_normalizer'] * 
        (1 + data['positive_divergence'] - data['negative_divergence'])
    )
    
    # Momentum-Efficiency Alignment Factor
    data['momentum_efficiency_factor'] = (
        data['momentum_efficiency_alignment'] * 
        data['efficiency_consistency'] * 
        data['confirmed_move']
    )
    
    # Final Regime-Adaptive Predictive Signal
    data['final_factor'] = (
        data['divergence_score'] + 
        data['momentum_efficiency_factor'] + 
        data['high_vol_signal'] + 
        data['low_vol_signal'] + 
        data['transition_signal']
    )
    
    # Apply regime-specific smoothing
    data['factor_smoothed'] = np.where(
        data['vol_regime'] == 'high_vol',
        data['final_factor'].rolling(window=2).mean(),
        np.where(
            data['vol_regime'] == 'low_vol',
            data['final_factor'].rolling(window=5).mean(),
            data['final_factor'].rolling(window=3).mean()
        )
    )
    
    return data['factor_smoothed']
