import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Regime Adjusted Volume Efficiency factor
    Combines multi-timeframe momentum analysis with volume efficiency metrics
    to generate regime-adaptive alpha signals
    """
    data = df.copy()
    
    # Multi-Timeframe Momentum Calculation
    data['momentum_5'] = data['close'] / data['close'].shift(5) - 1
    data['momentum_10'] = data['close'] / data['close'].shift(10) - 1
    data['momentum_20'] = data['close'] / data['close'].shift(20) - 1
    
    # Momentum convergence/divergence patterns
    data['momentum_convergence'] = (
        (data['momentum_5'] * data['momentum_10'] > 0) & 
        (np.abs(data['momentum_5'] - data['momentum_10']) < np.abs(data['momentum_5'] * 0.5))
    ).astype(int)
    
    # Momentum Regime Identification
    data['momentum_strength'] = (
        data['momentum_5'].rolling(window=10).mean() + 
        data['momentum_10'].rolling(window=10).mean()
    ) / 2
    
    # Momentum acceleration/deceleration
    data['momentum_accel_5'] = data['momentum_5'] - data['momentum_5'].shift(5)
    data['momentum_accel_10'] = data['momentum_10'] - data['momentum_10'].shift(10)
    
    # Classify momentum regimes
    conditions = [
        (data['momentum_strength'] > 0.02) & (data['momentum_accel_5'] > 0),
        (data['momentum_strength'] < -0.02) & (data['momentum_accel_5'] < 0),
        (np.abs(data['momentum_strength']) < 0.01) & (np.abs(data['momentum_accel_5']) < 0.005)
    ]
    choices = [2, -2, 0]  # Accelerating, Decelerating, Stable
    data['momentum_regime'] = np.select(conditions, choices, default=1)  # Default: Moderate
    
    # Volume Efficiency Metrics
    data['price_range'] = data['high'] - data['low']
    data['volume_efficiency'] = data['price_range'] / (data['volume'] + 1e-8)
    
    # Volume concentration at price extremes
    data['high_volume_ratio'] = (
        (data['close'] - data['low']) / (data['high'] - data['low'] + 1e-8) * 
        data['volume']
    )
    data['low_volume_ratio'] = (
        (data['high'] - data['close']) / (data['high'] - data['low'] + 1e-8) * 
        data['volume']
    )
    
    # Volume persistence patterns
    data['volume_trend'] = data['volume'].rolling(window=5).apply(
        lambda x: 1 if (x.diff().dropna() > 0).sum() >= 3 else -1 if (x.diff().dropna() < 0).sum() >= 3 else 0
    )
    
    # Regime-Specific Volume Analysis
    # Volume behavior during momentum acceleration
    data['accel_volume_confirmation'] = (
        (data['momentum_regime'] == 2) & 
        (data['volume_trend'] == 1) & 
        (data['high_volume_ratio'] > data['high_volume_ratio'].rolling(window=10).mean())
    ).astype(int)
    
    # Volume characteristics during momentum deceleration
    data['decel_volume_divergence'] = (
        (data['momentum_regime'] == -2) & 
        (data['volume_trend'] == -1) & 
        (data['low_volume_ratio'] > data['low_volume_ratio'].rolling(window=10).mean())
    ).astype(int)
    
    # Volume confirmation during regime transitions
    regime_changes = data['momentum_regime'].diff().fillna(0) != 0
    data['transition_volume_signal'] = (
        regime_changes & 
        (data['volume'] > data['volume'].rolling(window=20).mean()) &
        data['momentum_convergence']
    ).astype(int)
    
    # Generate Adaptive Alpha Signal
    # Base signal components
    momentum_signal = data['momentum_strength'] * (1 + data['momentum_convergence'])
    volume_efficiency_signal = data['volume_efficiency'].rolling(window=10).mean()
    
    # Regime-dependent scaling
    regime_weights = {
        2: 1.2,   # Accelerating momentum
        1: 1.0,   # Moderate momentum
        0: 0.5,   # Stable momentum
        -1: -1.0, # Moderate negative
        -2: -1.2  # Decelerating negative
    }
    
    data['regime_weight'] = data['momentum_regime'].map(regime_weights)
    
    # Enhanced signals during regime transitions
    transition_boost = data['transition_volume_signal'] * 0.3
    
    # Volume confirmation signals
    volume_confirmation = (
        data['accel_volume_confirmation'] * 0.2 - 
        data['decel_volume_divergence'] * 0.2
    )
    
    # Final alpha factor
    alpha_signal = (
        momentum_signal * data['regime_weight'] +
        volume_efficiency_signal * 0.5 +
        volume_confirmation +
        transition_boost
    )
    
    # Normalize and smooth the signal
    alpha_factor = alpha_signal.rolling(window=5).mean()
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=50).mean()) / alpha_factor.rolling(window=50).std()
    
    return alpha_factor
