import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Asymmetric Volatility Response Factor with Volume-Weighted Price Entropy
    Combines volatility asymmetry analysis with market information content measurement
    """
    # Make a copy to avoid modifying original data
    data = df.copy()
    
    # Calculate daily returns from open
    returns = (data['close'] - data['open']) / data['open']
    
    # 1. Asymmetric Volatility Response Factor
    # Calculate upside and downside volatility using 20-day rolling window
    upside_returns = returns[returns > 0]
    downside_returns = returns[returns < 0]
    
    # Rolling volatility calculations
    upside_vol = returns.rolling(window=20).apply(
        lambda x: np.std(x[x > 0]) if len(x[x > 0]) > 5 else np.nan, 
        raw=True
    )
    
    downside_vol = returns.rolling(window=20).apply(
        lambda x: np.std(x[x < 0]) if len(x[x < 0]) > 5 else np.nan, 
        raw=True
    )
    
    # Volatility ratio - measure of asymmetry
    vol_ratio = upside_vol / downside_vol
    
    # Normalize and handle extremes
    vol_ratio_zscore = (vol_ratio - vol_ratio.rolling(window=60).mean()) / vol_ratio.rolling(window=60).std()
    
    # 2. Volume-Weighted Price Entropy
    # Calculate price changes weighted by volume
    price_changes = data['close'].pct_change()
    volume_weighted_changes = price_changes * data['volume']
    
    # Calculate entropy using rolling window of 10 days
    def calculate_entropy(series):
        if len(series.dropna()) < 5:
            return np.nan
        # Discretize into bins for entropy calculation
        hist, _ = np.histogram(series.dropna(), bins=5, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        return -np.sum(hist * np.log(hist))
    
    entropy = volume_weighted_changes.rolling(window=10).apply(
        calculate_entropy, raw=False
    )
    
    # Normalize entropy
    entropy_zscore = (entropy - entropy.rolling(window=60).mean()) / entropy.rolling(window=60).std()
    
    # 3. Combine factors with regime awareness
    # High entropy suggests information events, low entropy suggests efficiency
    # Combine with volatility asymmetry for regime-dependent signal
    
    # Final factor: negative relationship between entropy and volatility asymmetry
    # High entropy + high vol asymmetry → potential regime change
    # Low entropy + stable vol ratio → trend continuation
    factor = -entropy_zscore * vol_ratio_zscore
    
    # Smooth the factor with 5-day moving average
    factor_smoothed = factor.rolling(window=5).mean()
    
    # Handle NaN values by forward filling (but not looking ahead)
    factor_smoothed = factor_smoothed.fillna(method='ffill')
    
    return factor_smoothed
