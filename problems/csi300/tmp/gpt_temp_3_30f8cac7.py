import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Momentum-Fractality Divergence Factor
    Combines multi-scale momentum divergence with fractal pattern recognition
    and volume-fractal congruence for regime-adaptive factor generation.
    """
    data = df.copy()
    
    # Multi-Scale Momentum Divergence
    # Short-term momentum: 5-day log return
    mom_short = np.log(data['close'] / data['close'].shift(5))
    
    # Medium-term momentum: 10-day log return  
    mom_medium = np.log(data['close'] / data['close'].shift(10))
    
    # Long-term momentum: 20-day log return
    mom_long = np.log(data['close'] / data['close'].shift(20))
    
    # Momentum divergence score
    momentum_divergence = pd.Series(0, index=data.index)
    for i in range(20, len(data)):
        window_short = mom_short.iloc[i-5:i+1]
        window_medium = mom_medium.iloc[i-5:i+1]
        window_long = mom_long.iloc[i-5:i+1]
        
        # Count days where short > medium > long or short < medium < long
        divergence_days = 0
        for j in range(len(window_short)):
            if (window_short.iloc[j] > window_medium.iloc[j] > window_long.iloc[j]) or \
               (window_short.iloc[j] < window_medium.iloc[j] < window_long.iloc[j]):
                divergence_days += 1
        momentum_divergence.iloc[i] = divergence_days / len(window_short)
    
    # Fractal Pattern Recognition - Hurst exponent approximation
    def hurst_approximation(series, window=20):
        """Approximate Hurst exponent using R/S analysis"""
        hurst_values = pd.Series(np.nan, index=series.index)
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i+1]
            returns = np.log(window_data / window_data.shift(1)).dropna()
            
            if len(returns) < 10:
                hurst_values.iloc[i] = 0.5
                continue
                
            # R/S analysis
            max_lag = min(10, len(returns)//2)
            rs_values = []
            for lag in range(2, max_lag+1):
                # Calculate rescaled range
                mean_return = returns.rolling(lag).mean().iloc[-1]
                deviations = returns.iloc[-lag:] - mean_return
                cumulative_deviations = deviations.cumsum()
                R = cumulative_deviations.max() - cumulative_deviations.min()
                S = deviations.std()
                if S > 0:
                    rs_values.append(R / S)
            
            if len(rs_values) > 1:
                # Fit log(R/S) vs log(lag)
                lags = np.log(range(2, len(rs_values)+2))
                rs_log = np.log(rs_values)
                hurst = np.polyfit(lags, rs_log, 1)[0]
                hurst_values.iloc[i] = hurst
            else:
                hurst_values.iloc[i] = 0.5
                
        return hurst_values.fillna(0.5)
    
    hurst_price = hurst_approximation(data['close'], window=20)
    
    # Fractal consistency across windows
    hurst_10 = hurst_approximation(data['close'], window=10)
    hurst_15 = hurst_approximation(data['close'], window=15)
    fractal_consistency = (hurst_price + hurst_10 + hurst_15) / 3
    
    # Volume-Fractal Congruence
    def volume_fractal_dimension(volume_series, window=10):
        """Calculate fractal dimension of volume series"""
        fd_values = pd.Series(np.nan, index=volume_series.index)
        for i in range(window, len(volume_series)):
            window_vol = volume_series.iloc[i-window:i+1]
            if window_vol.std() == 0:
                fd_values.iloc[i] = 1.0
                continue
                
            # Box-counting method approximation
            max_vol = window_vol.max()
            min_vol = window_vol.min()
            if max_vol == min_vol:
                fd_values.iloc[i] = 1.0
                continue
                
            box_sizes = [2, 4, 8]
            counts = []
            for box_size in box_sizes:
                box_count = np.ceil((max_vol - min_vol) / (window_vol.std() / box_size))
                counts.append(box_count)
            
            if len(counts) > 1:
                log_sizes = np.log([1/s for s in box_sizes])
                log_counts = np.log(counts)
                fd = -np.polyfit(log_sizes, log_counts, 1)[0]
                fd_values.iloc[i] = min(max(fd, 1.0), 2.0)
            else:
                fd_values.iloc[i] = 1.0
                
        return fd_values.fillna(1.0)
    
    vol_fractal = volume_fractal_dimension(data['volume'], window=10)
    
    # Price-volume fractal alignment
    price_vol_alignment = np.abs(hurst_price - vol_fractal)
    
    # Adaptive Momentum-Fractal Synthesis
    regime_weights = pd.Series(0.0, index=data.index)
    momentum_fractal_scores = pd.Series(0.0, index=data.index)
    
    for i in range(20, len(data)):
        # Regime classification
        if hurst_price.iloc[i] > 0.6:  # High persistence regime
            regime_weight = 0.7
            # Momentum convergence weighting
            mom_weight = 1.0 - momentum_divergence.iloc[i]
        elif hurst_price.iloc[i] < 0.4:  # Low persistence/anti-persistence regime
            regime_weight = 0.3
            # Momentum divergence weighting
            mom_weight = momentum_divergence.iloc[i]
        else:  # Random regime
            regime_weight = 0.5
            mom_weight = 0.5
            
        regime_weights.iloc[i] = regime_weight
        
        # Volume fractal confirmation
        vol_confirmation = 1.0 - price_vol_alignment.iloc[i]
        
        # Construct regime-specific score
        momentum_fractal_scores.iloc[i] = (mom_weight * regime_weight * 
                                          fractal_consistency.iloc[i] * 
                                          vol_confirmation)
    
    # Dynamic Regime Transition Factor
    transition_intensity = pd.Series(0.0, index=data.index)
    regime_persistence = pd.Series(1.0, index=data.index)
    
    for i in range(21, len(data)):
        # Detect regime transitions
        hurst_change = abs(hurst_price.iloc[i] - hurst_price.iloc[i-1])
        vol_change = abs(data['volume'].iloc[i] / data['volume'].iloc[i-1] - 1)
        price_accel = abs(mom_short.iloc[i] - mom_short.iloc[i-1])
        
        # Transition intensity
        transition_intensity.iloc[i] = (hurst_change + vol_change + price_accel) / 3
        
        # Regime persistence decay
        if transition_intensity.iloc[i] > 0.1:  # Significant transition
            regime_persistence.iloc[i] = regime_persistence.iloc[i-1] * 0.8
        else:
            regime_persistence.iloc[i] = min(regime_persistence.iloc[i-1] * 1.05, 1.0)
    
    # Final factor with transition-adaptive weights
    final_factor = (momentum_fractal_scores * regime_persistence * 
                   (1.0 - transition_intensity))
    
    # Normalize and clean
    final_factor = final_factor.replace([np.inf, -np.inf], np.nan).fillna(0)
    final_factor = (final_factor - final_factor.rolling(50, min_periods=20).mean()) / \
                   final_factor.rolling(50, min_periods=20).std()
    
    return final_factor.fillna(0)
