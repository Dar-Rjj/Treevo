import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Price Fractal Efficiency Factor
    Combines price efficiency and volume fractal analysis to detect market regimes
    and generate adaptive trading signals.
    """
    # Copy dataframe to avoid modifying original
    data = df.copy()
    
    # 1. Price Efficiency Component
    # True Range Calculation
    data['tr1'] = data['high'] - data['low']
    data['tr2'] = abs(data['high'] - data['close'].shift(1))
    data['tr3'] = abs(data['low'] - data['close'].shift(1))
    data['true_range'] = data[['tr1', 'tr2', 'tr3']].max(axis=1)
    
    # Cumulative True Range (5-day)
    data['cumulative_tr_5d'] = data['true_range'].rolling(window=5, min_periods=3).sum()
    
    # Net vs Absolute Movement (5-day)
    data['net_change_5d'] = data['close'] - data['close'].shift(5)
    data['abs_movement_5d'] = abs(data['close'] - data['close'].shift(1)).rolling(window=5, min_periods=3).sum()
    
    # Price Efficiency Ratio
    data['price_efficiency'] = np.where(
        data['abs_movement_5d'] > 0,
        abs(data['net_change_5d']) / data['abs_movement_5d'],
        0
    )
    
    # 2. Volume Fractal Component
    # Daily Volume Changes
    data['volume_change'] = data['volume'].pct_change()
    
    # Volume Volatility (20-day)
    data['volume_volatility'] = data['volume_change'].rolling(window=20, min_periods=10).std()
    
    # Rescaled Range Analysis for Fractal Dimension
    def calculate_hurst_exponent(series, max_lag=20):
        """Calculate Hurst exponent using rescaled range analysis"""
        if len(series) < max_lag:
            return 0.5
        
        lags = range(2, min(max_lag, len(series)))
        tau = []
        
        for lag in lags:
            # Create non-overlapping subseries
            subseries = [series.values[i:i+lag] for i in range(0, len(series)-lag, lag)]
            if len(subseries) < 2:
                continue
                
            # Calculate R/S for each subseries
            rs_values = []
            for sub in subseries:
                if len(sub) < 2:
                    continue
                mean_sub = np.mean(sub)
                deviations = sub - mean_sub
                cumulative_dev = np.cumsum(deviations)
                r = np.max(cumulative_dev) - np.min(cumulative_dev)
                s = np.std(sub)
                if s > 0:
                    rs_values.append(r / s)
            
            if rs_values:
                tau.append(np.log(np.mean(rs_values)))
        
        if len(tau) < 2:
            return 0.5
            
        # Fit linear regression to log-log plot
        lags_log = np.log(lags[:len(tau)])
        hurst = np.polyfit(lags_log, tau, 1)[0]
        return hurst
    
    # Calculate rolling Hurst exponent for volume
    hurst_values = []
    for i in range(len(data)):
        if i < 30:
            hurst_values.append(0.5)
            continue
        
        window_data = data['volume'].iloc[max(0, i-30):i+1]
        hurst = calculate_hurst_exponent(window_data)
        hurst_values.append(hurst)
    
    data['volume_hurst'] = hurst_values
    
    # 3. Integration & Signal Generation
    # Multi-timeframe analysis
    data['efficiency_3d'] = data['price_efficiency'].rolling(window=3, min_periods=2).mean()
    data['efficiency_10d'] = data['price_efficiency'].rolling(window=10, min_periods=5).mean()
    data['efficiency_20d'] = data['price_efficiency'].rolling(window=20, min_periods=10).mean()
    
    # Regime Detection
    data['trending_regime'] = (
        (data['volume_hurst'] > 0.65) & 
        (data['price_efficiency'] > 0.6)
    ).astype(int)
    
    data['mean_reverting_regime'] = (
        (data['volume_hurst'] < 0.35) & 
        (data['price_efficiency'] < 0.4)
    ).astype(int)
    
    # Signal Weighting
    data['trend_strength'] = (
        0.4 * data['efficiency_3d'] + 
        0.35 * data['efficiency_10d'] + 
        0.25 * data['efficiency_20d']
    )
    
    # Final Factor Calculation
    data['fractal_efficiency_factor'] = (
        data['trend_strength'] * 
        (1 + 0.5 * data['trending_regime'] - 0.3 * data['mean_reverting_regime']) * 
        (1.2 - data['volume_volatility'].clip(upper=1.0))
    )
    
    # Clean up intermediate columns
    result = data['fractal_efficiency_factor'].copy()
    
    return result
