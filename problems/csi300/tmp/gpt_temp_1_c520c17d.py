import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate returns
    data['returns'] = data['close'].pct_change()
    data['squared_returns'] = data['returns'] ** 2
    
    # 1. Fractal Efficiency Calculation
    # Path Length: Sum of absolute daily returns over 10 days
    data['path_length'] = data['returns'].abs().rolling(window=10).sum()
    
    # Net Distance: Absolute difference between current and 10-day prior close
    data['net_distance'] = (data['close'] - data['close'].shift(10)).abs()
    
    # Efficiency Ratio
    data['efficiency_ratio'] = data['net_distance'] / data['path_length']
    data['efficiency_ratio'] = data['efficiency_ratio'].clip(0, 1)  # Constrain to [0,1]
    
    # 2. Regime Detection
    # Volume Fractal Dimension (5-day window)
    data['volume_range'] = data['volume'].rolling(window=5).max() - data['volume'].rolling(window=5).min()
    data['volume_fractal'] = np.log(data['volume_range'] + 1e-8) / np.log(5)
    
    # Volume Regime Classification
    data['volume_regime'] = np.where(data['volume_fractal'] > data['volume_fractal'].rolling(window=20).median(), 1, -1)
    
    # Volatility Clustering (autocorrelation of squared returns, lag 1, 15-day window)
    def autocorr_squared_returns(series):
        if len(series) < 2:
            return 0
        return series.autocorr(lag=1)
    
    data['vol_clustering'] = data['squared_returns'].rolling(window=15).apply(
        autocorr_squared_returns, raw=False
    )
    
    # Volatility Regime
    data['vol_regime'] = np.where(data['vol_clustering'] > 0.1, 1, -1)
    
    # Combine Regime Signals
    data['combined_regime'] = data['volume_regime'] + data['vol_regime']
    data['regime_state'] = np.where(data['combined_regime'] > 0, 1, -1)
    
    # Regime Persistence (weight based on how long regime has persisted)
    data['regime_change'] = data['regime_state'] != data['regime_state'].shift(1)
    data['regime_persistence'] = data['regime_change'].rolling(window=10).apply(
        lambda x: 10 - np.sum(x), raw=False
    )
    data['regime_weight'] = data['regime_persistence'] / 10
    
    # 3. Adaptive Momentum Construction
    # Signed Efficiency (direction from daily return)
    data['return_sign'] = np.sign(data['close'] - data['close'].shift(1))
    data['signed_efficiency'] = data['efficiency_ratio'] * data['return_sign']
    
    # Regime-Dependent Scaling
    data['scaled_momentum'] = data['signed_efficiency'] * data['regime_weight']
    
    # Volume Confirmation Layer
    # Rolling correlation between volume and efficiency (8-day window)
    data['volume_efficiency_corr'] = data['volume'].rolling(window=8).corr(data['efficiency_ratio'])
    data['volume_efficiency_corr'] = data['volume_efficiency_corr'].fillna(0)
    
    # Dynamic Weighting based on correlation strength
    data['correlation_strength'] = data['volume_efficiency_corr'].abs()
    data['volume_weight'] = data['correlation_strength'] * data['regime_weight']
    
    # Final Alpha Generation
    # Combine components multiplicatively with regime smoothing
    data['alpha'] = data['scaled_momentum'] * data['volume_weight']
    
    # Apply regime transition smoothing
    regime_smooth_window = 3
    data['alpha_smoothed'] = data['alpha'].rolling(window=regime_smooth_window).mean()
    
    # Return the final alpha factor
    return data['alpha_smoothed']
