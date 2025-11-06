import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy import stats

def heuristics_v2(df):
    """
    Price-Volume Fractal Momentum with Regime Transition factor
    """
    data = df.copy()
    
    # Multi-timeframe Momentum Alignment
    # Calculate momentum across different timeframes
    mom_2d = data['close'] / data['close'].shift(2) - 1
    mom_5d = data['close'] / data['close'].shift(5) - 1
    mom_10d = data['close'] / data['close'].shift(10) - 1
    
    # Momentum convergence score
    positive_count = (mom_2d > 0).astype(int) + (mom_5d > 0).astype(int) + (mom_10d > 0).astype(int)
    negative_count = (mom_2d < 0).astype(int) + (mom_5d < 0).astype(int) + (mom_10d < 0).astype(int)
    momentum_convergence = positive_count - negative_count
    
    # Momentum regime strength (average absolute momentum)
    momentum_strength = (abs(mom_2d) + abs(mom_5d) + abs(mom_10d)) / 3
    
    # Fractal Market Structure Analysis
    def hurst_exponent(ts, window=20):
        """Calculate Hurst exponent using rescaled range analysis"""
        hurst_values = []
        for i in range(len(ts)):
            if i < window:
                hurst_values.append(np.nan)
                continue
                
            window_data = ts.iloc[i-window:i]
            if len(window_data) < 10:
                hurst_values.append(np.nan)
                continue
                
            # Calculate rescaled range for multiple time scales
            lags = range(2, min(10, len(window_data)//2))
            rs_values = []
            
            for lag in lags:
                # Create non-overlapping subsets
                subsets = [window_data.iloc[j:j+lag] for j in range(0, len(window_data)-lag+1, lag)]
                if len(subsets) < 2:
                    continue
                    
                # Calculate R/S for each subset
                subset_rs = []
                for subset in subsets:
                    if len(subset) < 2:
                        continue
                    mean_val = subset.mean()
                    deviations = subset - mean_val
                    cumulative_deviations = deviations.cumsum()
                    r = cumulative_deviations.max() - cumulative_deviations.min()
                    s = subset.std()
                    if s > 0:
                        subset_rs.append(r / s)
                
                if subset_rs:
                    rs_values.append(np.mean(subset_rs))
            
            if len(rs_values) > 1:
                # Log-log regression to estimate Hurst
                x = np.log(lags[:len(rs_values)])
                y = np.log(rs_values)
                if len(x) > 1:
                    slope, _, _, _, _ = stats.linregress(x, y)
                    hurst_values.append(slope)
                else:
                    hurst_values.append(np.nan)
            else:
                hurst_values.append(np.nan)
        
        return pd.Series(hurst_values, index=ts.index)
    
    # Calculate Hurst exponent
    hurst = hurst_exponent(data['close'], window=20)
    
    # Detect regime transitions
    hurst_rolling_mean = hurst.rolling(window=10, min_periods=5).mean()
    hurst_rolling_std = hurst.rolling(window=10, min_periods=5).std()
    
    regime_transition = np.zeros(len(data))
    for i in range(1, len(data)):
        if pd.notna(hurst.iloc[i]) and pd.notna(hurst.iloc[i-1]):
            if hurst_rolling_std.iloc[i] > 0:
                z_score = abs(hurst.iloc[i] - hurst_rolling_mean.iloc[i]) / hurst_rolling_std.iloc[i]
                if z_score > 1.5:
                    regime_transition[i] = z_score
    
    regime_transition = pd.Series(regime_transition, index=data.index)
    
    # Fractal complexity measure (volatility of Hurst exponent)
    fractal_complexity = hurst.rolling(window=10, min_periods=5).std()
    
    # Volume Confirmation Dynamics
    # Volume trend alignment
    volume_mom_5d = data['volume'] / data['volume'].shift(5) - 1
    price_mom_5d = data['close'] / data['close'].shift(5) - 1
    
    volume_price_concordance = np.sign(volume_mom_5d * price_mom_5d)
    
    # Volume shock analysis
    volume_20d_avg = data['volume'].rolling(window=20, min_periods=10).mean()
    volume_shock = (data['volume'] > 2 * volume_20d_avg).astype(int)
    
    # Shock absorption efficiency (price change relative to volume shock)
    shock_efficiency = np.zeros(len(data))
    for i in range(1, len(data)):
        if volume_shock.iloc[i] == 1:
            price_change = abs(data['close'].iloc[i] / data['close'].iloc[i-1] - 1)
            volume_change = data['volume'].iloc[i] / volume_20d_avg.iloc[i]
            if volume_change > 0:
                shock_efficiency[i] = price_change / volume_change
    
    shock_efficiency = pd.Series(shock_efficiency, index=data.index)
    
    # Volume persistence (autocorrelation of volume)
    volume_persistence = data['volume'].rolling(window=10, min_periods=5).apply(
        lambda x: x.autocorr() if len(x) > 1 else np.nan, raw=False
    )
    
    # Combine Momentum and Fractal Signals
    momentum_fractal_composite = (momentum_convergence * momentum_strength * 
                                 (1 - fractal_complexity.fillna(0)))
    
    # Volume confirmation as confidence weight
    volume_confidence = (0.5 + 0.5 * volume_price_concordance.fillna(0) * 
                        (1 + volume_persistence.fillna(0)))
    
    # Generate Regime-Adaptive Alpha Factor
    regime_adaptive_factor = (momentum_fractal_composite * 
                             volume_confidence * 
                             (1 + regime_transition.fillna(0)) * 
                             (1 - shock_efficiency.fillna(0)))
    
    return regime_adaptive_factor
