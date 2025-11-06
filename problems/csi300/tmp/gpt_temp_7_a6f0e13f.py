import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Price-Volume Fractal Divergence factor that detects divergences between 
    price and volume fractal properties to predict future returns.
    """
    data = df.copy()
    
    # Helper function for Hurst exponent calculation
    def hurst_exponent(series, window):
        """Calculate Hurst exponent using rescaled range analysis"""
        hurst_values = []
        for i in range(len(series) - window + 1):
            window_data = series.iloc[i:i+window]
            if len(window_data) < 2:
                hurst_values.append(np.nan)
                continue
            
            # Calculate mean and cumulative deviations
            mean_val = window_data.mean()
            deviations = window_data - mean_val
            cumulative_deviations = deviations.cumsum()
            
            # Calculate range and standard deviation
            R = cumulative_deviations.max() - cumulative_deviations.min()
            S = window_data.std()
            
            if S == 0:
                hurst_values.append(np.nan)
            else:
                hurst_values.append(R / S)
        
        # Pad with NaN to match original length
        hurst_padded = [np.nan] * (window - 1) + hurst_values
        return pd.Series(hurst_padded, index=series.index)
    
    # Calculate fractal dimensions for different windows
    def calculate_fractal_dimension(series, windows=[5, 10, 20]):
        """Calculate fractal dimension as 2 - Hurst exponent"""
        fractal_dims = pd.DataFrame(index=series.index)
        for window in windows:
            hurst = hurst_exponent(series, window)
            fractal_dim = 2 - hurst
            fractal_dims[f'fd_{window}'] = fractal_dim
        return fractal_dims.mean(axis=1)
    
    # Calculate efficiency ratio
    def efficiency_ratio(series, window=10):
        """Calculate efficiency ratio: net movement / total movement"""
        net_movement = abs(series - series.shift(window))
        total_movement = series.diff().abs().rolling(window=window).sum()
        efficiency = net_movement / total_movement
        return efficiency.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate volume clustering factor
    def volume_clustering_factor(volume_series, window=20):
        """Calculate persistence of volume regimes"""
        volume_median = volume_series.rolling(window=window).median()
        above_median = volume_series > volume_median
        
        # Count consecutive days in same regime
        clustering = []
        current_streak = 0
        current_state = False
        
        for i, state in enumerate(above_median):
            if i == 0 or state != current_state:
                current_streak = 1
                current_state = state
            else:
                current_streak += 1
            clustering.append(current_streak)
        
        clustering_series = pd.Series(clustering, index=volume_series.index)
        return clustering_series.rolling(window=window).mean()
    
    # Calculate large trade concentration
    def large_trade_concentration(amount, volume, window=20):
        """Calculate concentration of large trades"""
        trade_size = amount / volume.replace(0, np.nan)
        large_trade_threshold = trade_size.rolling(window=window).median() * 2
        
        # Identify large trades
        large_trades = trade_size > large_trade_threshold
        large_trade_amount = amount.where(large_trades, 0)
        
        # Concentration ratio
        concentration = large_trade_amount.rolling(window=window).sum() / \
                       amount.rolling(window=window).sum()
        return concentration.fillna(0)
    
    # Main calculations
    
    # Price fractal properties
    price_series = (data['high'] + data['low'] + data['close']) / 3
    price_fractal = calculate_fractal_dimension(price_series)
    price_efficiency = efficiency_ratio(data['close'])
    
    # Volume fractal properties
    volume_fractal = calculate_fractal_dimension(data['volume'])
    volume_clustering = volume_clustering_factor(data['volume'])
    large_trade_conc = large_trade_concentration(data['amount'], data['volume'])
    
    # Volume-weighted price fractal
    vwap = data['amount'] / data['volume'].replace(0, np.nan)
    vwap_fractal = calculate_fractal_dimension(vwap.fillna(method='ffill'))
    
    # Divergence calculations
    fractal_divergence = price_fractal - volume_fractal
    
    # Rolling correlation between price and volume fractal dimensions
    fractal_corr = price_fractal.rolling(window=20).corr(volume_fractal)
    
    # Efficiency divergence
    efficiency_divergence = price_efficiency - (1 - volume_clustering / volume_clustering.rolling(window=50).max())
    
    # Detect regime changes in fractal relationship
    fractal_regime_change = fractal_corr.diff().abs().rolling(window=5).mean()
    
    # Combine signals
    # Base divergence signal
    base_signal = fractal_divergence * efficiency_divergence
    
    # Weight by large trade concentration and regime stability
    weighted_signal = base_signal * large_trade_conc * (1 - fractal_regime_change)
    
    # Apply directional bias based on recent momentum
    price_momentum = data['close'].pct_change(5).rolling(window=5).mean()
    directional_bias = np.sign(price_momentum)
    
    # Final composite score
    composite_score = weighted_signal * directional_bias
    
    # Normalize the final factor
    factor = (composite_score - composite_score.rolling(window=50).mean()) / \
             composite_score.rolling(window=50).std()
    
    return factor.fillna(0)
