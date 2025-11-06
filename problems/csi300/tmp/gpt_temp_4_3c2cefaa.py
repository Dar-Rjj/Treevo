import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adaptive Price-Volume Divergence factor
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Volatility Regime Classification
    # Compute rolling volatility (20-day window)
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(window=20).std()
    
    # Compute volatility persistence (autocorrelation of volatility)
    vol_autocorr = []
    for i in range(len(data)):
        if i >= 40:  # Need enough data for autocorrelation
            vol_window = data['volatility'].iloc[i-20:i]
            if len(vol_window) >= 10 and vol_window.std() > 0:
                autocorr = vol_window.autocorr(lag=5)
                vol_autocorr.append(autocorr if not np.isnan(autocorr) else 0)
            else:
                vol_autocorr.append(0)
        else:
            vol_autocorr.append(0)
    
    data['vol_persistence'] = vol_autocorr
    
    # Identify volatility regime boundaries
    vol_ma = data['volatility'].rolling(window=10).mean()
    vol_std = data['volatility'].rolling(window=20).std()
    
    # High volatility regime: above mean + 0.5 std with high persistence
    high_vol_threshold = vol_ma + 0.5 * vol_std
    data['high_vol_regime'] = ((data['volatility'] > high_vol_threshold) & 
                              (data['vol_persistence'] > 0.3)).astype(int)
    
    # Low volatility regime: below mean with low persistence
    low_vol_threshold = vol_ma - 0.5 * vol_std
    data['low_vol_regime'] = ((data['volatility'] < low_vol_threshold) & 
                             (data['vol_persistence'] < 0.1)).astype(int)
    
    # 2. Regime-Specific Price-Volume Analysis
    # Price momentum and volume momentum
    data['price_momentum'] = data['close'] / data['close'].rolling(window=5).mean() - 1
    data['volume_momentum'] = data['volume'] / data['volume'].rolling(window=5).mean() - 1
    
    # Divergence in high volatility periods
    data['high_vol_divergence'] = data['price_momentum'] - data['volume_momentum']
    
    # Convergence in low volatility periods (inverse of divergence)
    data['low_vol_convergence'] = -abs(data['price_momentum'] - data['volume_momentum'])
    
    # 3. Multi-Frequency Volume Analysis
    # Extract volume periodic components using rolling windows
    data['volume_short'] = data['volume'].rolling(window=3).mean()
    data['volume_medium'] = data['volume'].rolling(window=8).mean()
    data['volume_long'] = data['volume'].rolling(window=21).mean()
    
    # Calculate volume spectral characteristics (ratio of frequencies)
    data['volume_spectral_ratio'] = (data['volume_short'] / data['volume_medium'] + 
                                   data['volume_medium'] / data['volume_long']) / 2
    
    # Volume trend consistency
    volume_trend = []
    for i in range(len(data)):
        if i >= 21:
            vol_window = data['volume'].iloc[i-21:i]
            if len(vol_window) >= 10:
                # Simple linear trend
                x = np.arange(len(vol_window))
                slope = np.polyfit(x, vol_window.values, 1)[0]
                volume_trend.append(slope / vol_window.mean() if vol_window.mean() > 0 else 0)
            else:
                volume_trend.append(0)
        else:
            volume_trend.append(0)
    
    data['volume_trend_strength'] = volume_trend
    
    # 4. Adaptive Signal Integration
    # Regime-conditional weighting
    regime_weight = []
    for i in range(len(data)):
        if data['high_vol_regime'].iloc[i] == 1:
            weight = 0.7  # Emphasize divergence in high volatility
        elif data['low_vol_regime'].iloc[i] == 1:
            weight = 0.3  # Emphasize convergence in low volatility
        else:
            weight = 0.5  # Neutral regime
            
        # Adjust weight based on volume spectral characteristics
        spectral_adjust = 1.0 + 0.2 * (data['volume_spectral_ratio'].iloc[i] - 1)
        regime_weight.append(weight * spectral_adjust)
    
    data['regime_weight'] = regime_weight
    
    # Combine divergence signals with volume features
    factor_values = []
    for i in range(len(data)):
        if i < 21:  # Need enough data for calculations
            factor_values.append(0)
            continue
            
        # Base signal from regime-appropriate measure
        if data['high_vol_regime'].iloc[i] == 1:
            base_signal = data['high_vol_divergence'].iloc[i]
        elif data['low_vol_regime'].iloc[i] == 1:
            base_signal = data['low_vol_convergence'].iloc[i]
        else:
            # Mixed regime - use weighted average
            base_signal = (0.6 * data['high_vol_divergence'].iloc[i] + 
                          0.4 * data['low_vol_convergence'].iloc[i])
        
        # Apply regime weighting
        weighted_signal = base_signal * data['regime_weight'].iloc[i]
        
        # Incorporate volume trend strength
        volume_adjusted = weighted_signal * (1 + 0.1 * data['volume_trend_strength'].iloc[i])
        
        # Final factor value with normalization
        if i >= 50:  # Use recent window for normalization
            recent_window = factor_values[-20:]
            if len(recent_window) > 0 and np.std(recent_window) > 0:
                normalized_value = (volume_adjusted - np.mean(recent_window)) / np.std(recent_window)
            else:
                normalized_value = volume_adjusted
        else:
            normalized_value = volume_adjusted
            
        factor_values.append(normalized_value)
    
    # Create output series
    factor_series = pd.Series(factor_values, index=data.index)
    
    return factor_series
