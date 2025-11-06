import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Novel alpha factor combining fractal market structure, behavioral momentum asymmetry,
    and non-linear price acceleration concepts using only current and historical data.
    """
    # Make a copy to avoid modifying original dataframe
    data = df.copy()
    
    # Fractal Market Structure - Multi-scale pattern identification
    # Calculate Hurst exponent approximation using different time scales
    def hurst_approximation(series, max_lag=20):
        """Approximate Hurst exponent using rescaled range analysis"""
        lags = range(2, min(max_lag + 1, len(series)))
        tau = []
        for lag in lags:
            # Create non-overlapping series
            series_lag = []
            for i in range(0, len(series) - lag, lag):
                series_lag.append(series.iloc[i + lag] - series.iloc[i])
            
            if len(series_lag) > 1:
                # Calculate mean and standard deviation
                mean_val = np.mean(series_lag)
                std_val = np.std(series_lag)
                if std_val > 0:
                    rs = (max(series_lag) - min(series_lag)) / std_val
                    tau.append(np.log(rs))
        
        if len(tau) > 1:
            # Linear regression to estimate Hurst exponent
            lags_log = np.log(lags[:len(tau)])
            hurst = np.polyfit(lags_log, tau, 1)[0]
            return hurst
        return 0.5
    
    # Calculate rolling Hurst exponent for fractal dimension
    hurst_values = []
    for i in range(len(data)):
        if i >= 50:  # Use 50-day window for Hurst calculation
            window = data['close'].iloc[i-50:i+1]
            hurst = hurst_approximation(window)
            hurst_values.append(hurst)
        else:
            hurst_values.append(0.5)
    
    data['hurst'] = hurst_values
    
    # Behavioral Momentum Asymmetry - Gain/loss response differential
    # Calculate asymmetric momentum based on recent price movements
    data['returns_5d'] = data['close'].pct_change(5)
    data['returns_10d'] = data['close'].pct_change(10)
    
    # Asymmetric momentum: stronger weight on gains than losses
    def asymmetric_momentum(returns_short, returns_long):
        """Calculate momentum with asymmetric response to gains vs losses"""
        gain_momentum = np.where(returns_short > 0, returns_short * 1.5, returns_short * 0.7)
        loss_momentum = np.where(returns_long < 0, returns_long * 0.8, returns_long * 1.2)
        return gain_momentum + loss_momentum
    
    data['behavioral_momentum'] = asymmetric_momentum(data['returns_5d'], data['returns_10d'])
    
    # Non-Linear Price Acceleration - Higher-order momentum dynamics
    # Calculate second derivative of price (acceleration)
    data['momentum_1'] = data['close'].pct_change(1)
    data['momentum_3'] = data['close'].pct_change(3)
    data['momentum_5'] = data['close'].pct_change(5)
    
    # Price acceleration as change in momentum
    data['acceleration_3_1'] = data['momentum_3'] - data['momentum_1']
    data['acceleration_5_3'] = data['momentum_5'] - data['momentum_3']
    
    # Non-linear acceleration factor using exponential weighting
    def non_linear_acceleration(acc1, acc2, hurst):
        """Calculate non-linear acceleration with fractal adjustment"""
        # Higher weight for consistent acceleration patterns
        consistency = np.sign(acc1) == np.sign(acc2)
        base_acc = (acc1 * 0.6 + acc2 * 0.4)
        
        # Adjust based on market regime (fractal dimension)
        if hurst > 0.6:  # Trending market
            return base_acc * (1 + hurst - 0.6)
        elif hurst < 0.4:  # Mean-reverting market
            return base_acc * (1 - (0.4 - hurst))
        else:  # Random walk
            return base_acc * 0.8
    
    # Calculate the combined alpha factor
    alpha_values = []
    for i in range(len(data)):
        if i >= 10:  # Ensure we have enough data
            hurst = data['hurst'].iloc[i]
            behavioral = data['behavioral_momentum'].iloc[i]
            acc1 = data['acceleration_3_1'].iloc[i] if not pd.isna(data['acceleration_3_1'].iloc[i]) else 0
            acc2 = data['acceleration_5_3'].iloc[i] if not pd.isna(data['acceleration_5_3'].iloc[i]) else 0
            
            nl_acc = non_linear_acceleration(acc1, acc2, hurst)
            
            # Combine factors with weights
            alpha = (hurst * 0.3 + behavioral * 0.4 + nl_acc * 0.3)
            alpha_values.append(alpha)
        else:
            alpha_values.append(0)
    
    # Create output series
    alpha_series = pd.Series(alpha_values, index=data.index)
    
    # Normalize the factor using rolling z-score (21-day window)
    def rolling_zscore(series, window=21):
        """Calculate rolling z-score"""
        rolling_mean = series.rolling(window=window, min_periods=1).mean()
        rolling_std = series.rolling(window=window, min_periods=1).std()
        return (series - rolling_mean) / rolling_std
    
    alpha_normalized = rolling_zscore(alpha_series)
    
    return alpha_normalized
