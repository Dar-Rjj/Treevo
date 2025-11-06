import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(df):
    # Multi-timeframe Momentum Alignment
    # Short-term Momentum (1-3 days)
    ret_1d = df['close'].pct_change(1)
    ret_3d = df['close'].pct_change(3)
    
    # Weighted combination with decay
    short_momentum = 0.6 * ret_1d + 0.4 * ret_3d
    
    # Medium-term Momentum (5-10 days)
    ret_5d = df['close'].pct_change(5)
    ret_10d = df['close'].pct_change(10)
    
    # Exponential decay weighting
    medium_momentum = 0.7 * ret_5d + 0.3 * ret_10d
    
    # Momentum Alignment Signal
    momentum_alignment = short_momentum.rolling(window=10, min_periods=5).corr(medium_momentum)
    
    # Combined momentum with alignment strength
    base_momentum = 0.5 * short_momentum + 0.5 * medium_momentum
    momentum_signal = base_momentum * (1 + momentum_alignment.fillna(0))
    
    # Volume Acceleration Confirmation
    def calculate_slope(series, window):
        slopes = pd.Series(index=series.index, dtype=float)
        for i in range(window-1, len(series)):
            if i >= window-1:
                y = series.iloc[i-window+1:i+1].values
                x = np.arange(len(y))
                if len(y) >= 2 and not np.all(y == y[0]):
                    slope, _, _, _, _ = linregress(x, y)
                    slopes.iloc[i] = slope
        return slopes
    
    # Recent volume slope (4-day)
    recent_volume_slope = calculate_slope(df['volume'], 4)
    
    # Historical volume slope (20-day)
    historical_volume_slope = calculate_slope(df['volume'], 20)
    
    # Volume acceleration
    volume_acceleration = recent_volume_slope - historical_volume_slope
    
    # Volume-Momentum Interaction
    momentum_volume_composite = momentum_signal * volume_acceleration.fillna(0)
    
    # Volatility-Normalized Returns
    # Daily volatility
    daily_volatility = (df['high'] - df['low']) / df['close']
    
    # Volatility regime
    vol_20d_median = daily_volatility.rolling(window=20, min_periods=10).median()
    vol_regime = (daily_volatility > vol_20d_median).astype(int)
    
    # Regime-adjusted volatility scaling
    regime_scaling = np.where(vol_regime == 1, 1.5, 1.0)
    adjusted_volatility = daily_volatility * regime_scaling
    
    # Risk-Adjusted Momentum
    risk_adjusted_momentum = momentum_volume_composite / (adjusted_volatility + 1e-8)
    
    # Regime-Aware Factor Combination
    # Market Regime Detection
    volume_regime = (volume_acceleration > volume_acceleration.rolling(window=20, min_periods=10).median()).astype(int)
    momentum_regime = (momentum_alignment > momentum_alignment.rolling(window=20, min_periods=10).median()).astype(int)
    
    # Adaptive Weighting
    def calculate_regime_weights(vol_regime, volume_regime, momentum_regime):
        weights = pd.DataFrame(index=vol_regime.index)
        
        # Base weights
        weights['momentum'] = 0.4
        weights['volume'] = 0.3
        weights['volatility'] = 0.3
        
        # Adjust weights based on regimes
        # High volatility: emphasize volatility normalization
        weights.loc[vol_regime == 1, 'volatility'] *= 1.5
        weights.loc[vol_regime == 1, 'momentum'] *= 0.7
        weights.loc[vol_regime == 1, 'volume'] *= 0.8
        
        # High volume acceleration: emphasize volume confirmation
        weights.loc[volume_regime == 1, 'volume'] *= 1.5
        weights.loc[volume_regime == 1, 'momentum'] *= 1.2
        
        # Strong momentum alignment: emphasize momentum component
        weights.loc[momentum_regime == 1, 'momentum'] *= 1.5
        weights.loc[momentum_regime == 1, 'volume'] *= 1.1
        
        # Normalize weights to sum to 1
        weight_sum = weights.sum(axis=1)
        weights = weights.div(weight_sum, axis=0)
        
        return weights
    
    regime_weights = calculate_regime_weights(vol_regime, volume_regime, momentum_regime)
    
    # Final factor combination
    momentum_component = momentum_signal.fillna(0)
    volume_component = volume_acceleration.fillna(0)
    volatility_component = (1 / (adjusted_volatility + 1e-8)).fillna(0)
    
    # Normalize components
    momentum_component_norm = (momentum_component - momentum_component.rolling(window=20, min_periods=10).mean()) / (momentum_component.rolling(window=20, min_periods=10).std() + 1e-8)
    volume_component_norm = (volume_component - volume_component.rolling(window=20, min_periods=10).mean()) / (volume_component.rolling(window=20, min_periods=10).std() + 1e-8)
    volatility_component_norm = (volatility_component - volatility_component.rolling(window=20, min_periods=10).mean()) / (volatility_component.rolling(window=20, min_periods=10).std() + 1e-8)
    
    # Apply regime-aware weighting
    final_factor = (regime_weights['momentum'] * momentum_component_norm + 
                   regime_weights['volume'] * volume_component_norm + 
                   regime_weights['volatility'] * volatility_component_norm)
    
    return final_factor
