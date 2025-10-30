import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Regime-Adaptive Multi-Scale Momentum Elasticity with Microstructure Enhancement
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Elasticity
    data['momentum_short'] = data['close'] / data['close'].shift(3) - 1
    data['momentum_medium'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_long'] = data['close'] / data['close'].shift(20) - 1
    data['momentum_elasticity'] = data['momentum_short'] * data['momentum_medium'] * data['momentum_long']
    
    # Gap Analysis and Reversion Context
    data['gap_magnitude'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)
    data['gap_avg_4d'] = data['gap_magnitude'].abs().rolling(window=4, min_periods=1).mean()
    data['gap_deviation'] = data['gap_magnitude'] / (data['gap_avg_4d'] + 1e-8)
    
    # Gap persistence (directional consistency over 3 days)
    gap_direction = np.sign(data['gap_magnitude'])
    data['gap_persistence'] = gap_direction.rolling(window=3, min_periods=1).apply(
        lambda x: len(set(x.dropna())) if len(x.dropna()) > 0 else 1
    )
    data['gap_persistence'] = 1 / (data['gap_persistence'] + 1e-8)  # Higher for more consistent directions
    
    data['reversion_pressure'] = data['close'] / data['close'].rolling(window=7, min_periods=1).mean() - 1
    
    # Microstructure and Volume-Price Dynamics
    # Market Microstructure Signals
    data['effective_spread'] = 2 * (data['close'] - (data['high'] + data['low']) / 2) / (data['high'] + data['low'] + 1e-8)
    
    # True Range calculation
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            abs(data['high'] - data['close'].shift(1)),
            abs(data['low'] - data['close'].shift(1))
        )
    )
    data['price_inefficiency'] = data['true_range'] / (abs(data['close'] - data['close'].shift(1)) + 1e-8)
    
    data['intraday_strength'] = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
    
    # Volume Dynamics Assessment
    data['volume_efficiency'] = data['volume'] / data['volume'].rolling(window=19, min_periods=1).mean()
    data['volume_deceleration'] = data['volume'].shift(5) / (data['volume'] + 1e-8) - 1
    
    volume_change = data['volume'] - data['volume'].shift(1)
    price_change_direction = np.sign(data['close'] - data['close'].shift(1))
    data['volume_price_divergence'] = price_change_direction * volume_change
    
    # Range Inefficiency Analysis
    data['daily_inefficiency'] = (data['high'] - data['low']) / (abs(data['close'] - data['open']) + 1e-8)
    data['range_efficiency'] = data['true_range'] / (data['high'] - data['low'] + 1e-8)
    
    # Inefficiency persistence (5-day directional inconsistency)
    inefficiency_direction = np.sign(data['daily_inefficiency'] - data['daily_inefficiency'].shift(1))
    data['inefficiency_persistence'] = inefficiency_direction.rolling(window=5, min_periods=1).apply(
        lambda x: len(set(x.dropna())) if len(x.dropna()) > 0 else 1
    )
    data['inefficiency_persistence'] = 1 / (data['inefficiency_persistence'] + 1e-8)
    
    # Volatility Regime Adaptation
    # Multi-Frequency Volatility Assessment
    returns = data['close'] / data['close'].shift(1) - 1
    data['vol_5d'] = returns.rolling(window=5, min_periods=1).std()
    data['vol_10d_range'] = ((data['high'] - data['low']) / data['close']).rolling(window=10, min_periods=1).mean()
    data['vol_15d'] = returns.rolling(window=15, min_periods=1).std()
    
    # Regime Classification
    data['volatility_state'] = data['vol_15d']
    data['vol_hist_avg'] = data['volatility_state'].rolling(window=29, min_periods=1).mean()
    data['historical_context'] = data['volatility_state'] / (data['vol_hist_avg'] + 1e-8)
    
    # Regime identification
    data['regime'] = 'transition'
    data.loc[data['historical_context'] < 0.7, 'regime'] = 'low'
    data.loc[data['historical_context'] > 1.5, 'regime'] = 'high'
    
    # Regime Transition Detection
    data['volatility_trajectory'] = data['volatility_state'] / (data['volatility_state'].shift(5) + 1e-8)
    data['recent_regime_change'] = abs(data['historical_context'] - data['historical_context'].shift(5))
    data['transition_scaling'] = 1 / (1 + abs(data['historical_context'] - 1))
    
    # Component Groups
    data['gap_momentum_core'] = data['gap_deviation'] * data['momentum_elasticity'] * data['gap_persistence']
    data['microstructure_enhancement'] = data['effective_spread'] * data['price_inefficiency'] * data['intraday_strength']
    data['volume_confirmation'] = data['volume_efficiency'] * data['volume_price_divergence'] * data['volume_deceleration']
    data['range_dynamics'] = data['daily_inefficiency'] * data['range_efficiency'] * data['inefficiency_persistence']
    
    # Regime-Adaptive Weighting
    high_vol_weight = np.where(data['regime'] == 'high', 1.0, 0.0)
    low_vol_weight = np.where(data['regime'] == 'low', 1.0, 0.0)
    transition_weight = np.where(data['regime'] == 'transition', 1.0, 0.0)
    
    # High Volatility Regime: Focus on momentum continuation
    high_vol_signal = (
        data['gap_momentum_core'] * 0.3 +
        data['microstructure_enhancement'] * 0.4 +
        data['volume_confirmation'] * 0.2 +
        data['range_dynamics'] * 0.1
    ) * data['volatility_trajectory']
    
    # Low Volatility Regime: Focus on gap reversion
    low_vol_signal = (
        data['gap_momentum_core'] * 0.5 +
        data['microstructure_enhancement'] * 0.2 +
        data['volume_confirmation'] * 0.2 +
        data['range_dynamics'] * 0.1
    ) * data['gap_persistence']
    
    # Transition Regime: Balanced approach
    transition_signal = (
        data['gap_momentum_core'] * 0.25 +
        data['microstructure_enhancement'] * 0.25 +
        data['volume_confirmation'] * 0.25 +
        data['range_dynamics'] * 0.25
    ) * data['transition_scaling'] * data['volatility_trajectory']
    
    # Final Composite Signal
    data['alpha_factor'] = (
        high_vol_weight * high_vol_signal +
        low_vol_weight * low_vol_signal +
        transition_weight * transition_signal
    )
    
    return data['alpha_factor']
