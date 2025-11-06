import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Cross-Association Momentum Divergence with Regime-Specific Acceleration
    """
    data = df.copy()
    
    # Price Momentum Divergence
    data['price_accel_short'] = (data['close'] / data['close'].shift(3)) - (data['close'].shift(3) / data['close'].shift(6))
    data['price_accel_medium'] = (data['close'] / data['close'].shift(10)) - (data['close'].shift(10) / data['close'].shift(20))
    
    # Volume-Price Association Strength
    data['volume_alignment'] = np.sign(data['close'] - data['close'].shift(1)) * np.sign(data['volume'] - data['volume'].shift(1))
    data['volume_persistence'] = np.abs(data['volume'] - data['volume'].shift(1)) / data['volume'].shift(1).replace(0, np.nan)
    
    # Cross-Association Scoring
    data['divergence_convergence'] = data['price_accel_short'] * data['volume_alignment']
    
    # Directional agreement frequency over 5-day window
    data['directional_agreement'] = data['volume_alignment'].rolling(window=5, min_periods=3).mean()
    
    # Magnitude coherence between price changes and volume spikes
    data['price_change_magnitude'] = np.abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)
    data['magnitude_coherence'] = data['price_change_magnitude'] * data['volume_persistence']
    
    # Volatility Regime Detection
    data['high'] = data['high'].fillna(method='ffill')
    data['low'] = data['low'].fillna(method='ffill')
    data['true_range'] = np.maximum(
        data['high'] - data['low'],
        np.maximum(
            np.abs(data['high'] - data['close'].shift(1)),
            np.abs(data['low'] - data['close'].shift(1))
        )
    )
    data['volatility_clustering'] = data['true_range'] / data['true_range'].shift(1).replace(0, np.nan)
    
    # Volatility persistence indicator
    volatility_direction = np.sign(data['volatility_clustering'] - 1)
    data['volatility_persistence'] = volatility_direction.groupby(
        (volatility_direction != volatility_direction.shift(1)).cumsum()
    ).cumcount() + 1
    
    # Volume Regime Identification
    data['volume_percentile'] = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    data['volume_shock'] = (data['volume'] > 2 * data['volume'].shift(5)).astype(int)
    
    # Regime-Adaptive Acceleration
    data['volatility_accel_factor'] = 1 + np.abs(data['volatility_clustering'] - 1)
    data['volume_accel_factor'] = data['volume_percentile'] / 0.5
    
    # Cross-Regime Interaction
    high_vol_regime = data['volatility_clustering'] > data['volatility_clustering'].rolling(window=20).quantile(0.7)
    high_vol_regime = high_vol_regime.fillna(False)
    high_volume_regime = data['volume_percentile'] > 0.7
    high_volume_regime = high_volume_regime.fillna(False)
    
    data['regime_multiplier'] = 1.0
    data.loc[high_vol_regime & high_volume_regime, 'regime_multiplier'] = 1.5
    data.loc[~high_vol_regime & (data['volume_percentile'] <= 0.7), 'regime_multiplier'] = 0.8
    
    # Multi-Timeframe Divergence Signals
    data['short_term_divergence'] = data['price_accel_short'] * data['volume_alignment'] * data['regime_multiplier']
    
    # Medium-term divergence momentum
    data['medium_term_divergence'] = data['price_accel_medium'] * data['directional_agreement'].rolling(window=10, min_periods=5).mean()
    
    # Divergence Persistence Scoring
    data['divergence_persistence'] = data['short_term_divergence'].rolling(window=10, min_periods=5).apply(
        lambda x: len([i for i in range(1, len(x)) if np.sign(x.iloc[i]) == np.sign(x.iloc[i-1])]) / max(1, len(x)-1),
        raw=False
    )
    
    # Cross-Association Factor Synthesis
    data['core_divergence_signal'] = (
        data['short_term_divergence'] * 0.6 + 
        data['medium_term_divergence'] * 0.4
    ) * data['divergence_persistence']
    
    # Final alpha factor with regime acceleration
    data['alpha_factor'] = (
        data['core_divergence_signal'] * 
        data['volatility_accel_factor'] * 
        data['volume_accel_factor'] * 
        data['magnitude_coherence'].rolling(window=5, min_periods=3).mean()
    )
    
    # Clean and return
    alpha_series = data['alpha_factor'].replace([np.inf, -np.inf], np.nan).fillna(method='ffill')
    return alpha_series
