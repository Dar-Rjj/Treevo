import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volume-Price Fractal Divergence Alpha Factor
    Combines fractal analysis of price and volume patterns to detect divergence signals
    """
    
    # Calculate price fractal dimension using high-low range complexity
    def calculate_price_fractal(high, low, close, window=20):
        # High-low range complexity
        hl_range = high - low
        range_volatility = hl_range.rolling(window=window).std()
        range_mean = hl_range.rolling(window=window).mean()
        range_complexity = range_volatility / (range_mean + 1e-8)
        
        # Close price path length (Hurst-like measure)
        close_returns = close.pct_change()
        close_volatility = close_returns.rolling(window=window).std()
        close_path = close.diff().abs().rolling(window=window).sum()
        close_complexity = close_path / (close_volatility * np.sqrt(window) + 1e-8)
        
        # Combined price fractal dimension
        price_fractal = 0.6 * range_complexity + 0.4 * close_complexity
        return price_fractal
    
    # Calculate volume fractal dimension
    def calculate_volume_fractal(volume, window=20):
        # Volume pattern irregularity
        volume_returns = volume.pct_change()
        volume_volatility = volume_returns.rolling(window=window).std()
        volume_mean = volume.rolling(window=window).mean()
        volume_irregularity = volume_volatility / (volume_mean + 1e-8)
        
        # Volume sequence complexity (autocorrelation breakdown)
        volume_autocorr = volume.rolling(window=window).apply(
            lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
        )
        volume_complexity = 1 - abs(volume_autocorr.fillna(0))
        
        # Combined volume fractal dimension
        volume_fractal = 0.7 * volume_irregularity + 0.3 * volume_complexity
        return volume_fractal
    
    # Detect divergence patterns
    def detect_divergence(price_fractal, volume_fractal, window=5):
        # Fractal dimension comparison
        price_gt_volume = (price_fractal > volume_fractal).astype(int)
        volume_gt_price = (volume_fractal > price_fractal).astype(int)
        
        # Divergence persistence
        price_divergence_persistence = price_gt_volume.rolling(window=window).sum()
        volume_divergence_persistence = volume_gt_price.rolling(window=window).sum()
        
        # Net divergence signal
        divergence_signal = price_divergence_persistence - volume_divergence_persistence
        
        return divergence_signal
    
    # Calculate all components
    price_fractal = calculate_price_fractal(df['high'], df['low'], df['close'])
    volume_fractal = calculate_volume_fractal(df['volume'])
    
    # Generate final alpha factor
    alpha_factor = detect_divergence(price_fractal, volume_fractal)
    
    # Normalize the factor
    alpha_factor = (alpha_factor - alpha_factor.rolling(window=50).mean()) / (alpha_factor.rolling(window=50).std() + 1e-8)
    
    return alpha_factor
