import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress

def heuristics_v2(data):
    """
    Generate alpha factor combining momentum-adjusted volume breakout with volatility-regime adaptation.
    """
    df = data.copy()
    
    # Momentum-Adjusted Volume Breakout
    # Calculate Recent Price Momentum using linear regression slope
    momentum_window = 10
    momentum_slopes = []
    for i in range(len(df)):
        if i < momentum_window - 1:
            momentum_slopes.append(np.nan)
        else:
            window_data = df['close'].iloc[i-momentum_window+1:i+1].values
            x = np.arange(len(window_data))
            slope, _, _, _, _ = linregress(x, window_data)
            momentum_slopes.append(slope)
    
    df['momentum'] = momentum_slopes
    
    # Volume Breakout Events
    volume_window = 20
    df['volume_ma'] = df['volume'].rolling(window=volume_window, min_periods=1).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Combine momentum with volume breakout
    df['momentum_volume'] = df['momentum'] * df['volume_ratio']
    df['momentum_volume_signal'] = np.sign(df['momentum_volume'])
    
    # Volatility-Regime Adaptive Factor
    # Short-Term Volatility using High-Low range
    volatility_window = 5
    df['daily_range'] = df['high'] - df['low']
    df['volatility_atr'] = df['daily_range'].rolling(window=volatility_window, min_periods=1).mean()
    
    # Price Efficiency using entropy (simplified)
    # Since we don't have intraday 15-minute data, use daily returns entropy
    df['daily_return'] = df['close'].pct_change()
    entropy_window = 10
    
    def calculate_entropy(returns_series):
        if len(returns_series) < 2:
            return np.nan
        # Discretize returns into bins
        bins = np.linspace(returns_series.min(), returns_series.max(), 6)
        digitized = np.digitize(returns_series, bins)
        counts = np.bincount(digitized)
        probs = counts[counts > 0] / len(returns_series)
        entropy = -np.sum(probs * np.log(probs))
        return entropy
    
    df['price_efficiency'] = df['daily_return'].rolling(window=entropy_window).apply(
        calculate_entropy, raw=False
    )
    
    # Combine volatility and efficiency
    df['vol_efficiency_ratio'] = df['price_efficiency'] / (df['volatility_atr'] + 1e-8)
    df['log_vol_efficiency'] = np.log1p(np.abs(df['vol_efficiency_ratio'])) * np.sign(df['vol_efficiency_ratio'])
    
    # Final alpha factor combining both components
    df['alpha_factor'] = (
        df['momentum_volume_signal'] * 0.6 + 
        df['log_vol_efficiency'] * 0.4
    )
    
    # Clean up intermediate columns
    result = df['alpha_factor'].copy()
    
    return result
