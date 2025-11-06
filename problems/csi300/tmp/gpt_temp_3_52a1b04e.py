import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import linregress, entropy
from sklearn.preprocessing import StandardScaler

def heuristics_v2(df):
    """
    Hierarchical Fractal Efficiency Momentum with Volume-Entropy Informed Reversal
    Combines fractal efficiency analysis with volume entropy to detect momentum and reversal patterns
    """
    df = df.copy()
    
    # 1. Hierarchical Fractal Efficiency Momentum
    # Calculate fractal dimension using multiple time windows
    def fractal_dimension(series, window):
        """Calculate fractal dimension using box counting method"""
        if len(series) < window:
            return np.nan
        
        # Normalize the series
        normalized = (series - series.min()) / (series.max() - series.min() + 1e-8)
        
        # Calculate range and standard deviation
        price_range = normalized.max() - normalized.min()
        std_dev = normalized.std()
        
        if std_dev == 0 or price_range == 0:
            return 1.0
        
        # Simplified fractal dimension estimation
        fractal_dim = 1 + np.log(std_dev + 1) / np.log(price_range * len(series) + 1)
        return min(max(fractal_dim, 1.0), 2.0)
    
    # Calculate fractal efficiency (1/fractal dimension)
    windows = [5, 10, 20]
    fractal_efficiency = pd.Series(index=df.index, dtype=float)
    
    for i in range(max(windows), len(df)):
        current_data = df['close'].iloc[i-max(windows):i+1]
        efficiencies = []
        for window in windows:
            if len(current_data) >= window:
                window_data = current_data[-window:]
                fd = fractal_dimension(window_data, window)
                eff = 1.0 / fd if fd > 0 else 1.0
                efficiencies.append(eff)
        
        if efficiencies:
            fractal_efficiency.iloc[i] = np.mean(efficiencies)
    
    # Multi-timeframe momentum congruence
    def momentum_congruence(close_prices, short_window=5, medium_window=10, long_window=20):
        """Calculate momentum congruence across multiple timeframes"""
        if len(close_prices) < long_window:
            return 0.0
        
        short_ma = close_prices.rolling(window=short_window).mean()
        medium_ma = close_prices.rolling(window=medium_window).mean()
        long_ma = close_prices.rolling(window=long_window).mean()
        
        # Calculate momentum directions
        short_trend = (short_ma > short_ma.shift(1)).astype(int)
        medium_trend = (medium_ma > medium_ma.shift(1)).astype(int)
        long_trend = (long_ma > long_ma.shift(1)).astype(int)
        
        # Calculate congruence (alignment of trends)
        congruence = (short_trend + medium_trend + long_trend) / 3.0
        return congruence.iloc[-1] if not congruence.empty else 0.0
    
    momentum_scores = pd.Series(index=df.index, dtype=float)
    for i in range(20, len(df)):
        momentum_scores.iloc[i] = momentum_congruence(df['close'].iloc[:i+1])
    
    # 2. Volume-Entropy Informed Reversal
    def volume_entropy(volume_series, window=10):
        """Calculate information entropy of volume distribution"""
        if len(volume_series) < window:
            return 0.5
        
        window_data = volume_series.iloc[-window:]
        
        # Create histogram of volume distribution
        hist, _ = np.histogram(window_data, bins=min(5, len(window_data)), density=True)
        hist = hist[hist > 0]  # Remove zero bins
        
        if len(hist) <= 1:
            return 0.5
        
        # Calculate entropy and normalize
        ent = entropy(hist)
        max_entropy = np.log(len(hist))
        normalized_entropy = ent / max_entropy if max_entropy > 0 else 0.5
        
        return normalized_entropy
    
    def price_reversal_strength(close_prices, volume_entropy_val, window=10):
        """Detect price reversal strength using entropy extremes"""
        if len(close_prices) < window:
            return 0.0
        
        # Calculate price momentum
        returns = close_prices.pct_change().iloc[-window:]
        current_return = returns.iloc[-1] if not returns.empty else 0
        
        # High entropy suggests chaotic volume distribution (potential reversal)
        # Low entropy suggests concentrated volume (momentum continuation)
        if volume_entropy_val > 0.7:  # High entropy - look for reversals
            reversal_strength = -current_return * volume_entropy_val
        elif volume_entropy_val < 0.3:  # Low entropy - momentum continuation
            reversal_strength = current_return * (1 - volume_entropy_val)
        else:  # Moderate entropy - neutral
            reversal_strength = current_return * 0.5
        
        return reversal_strength
    
    # Calculate volume entropy and reversal strength
    volume_entropy_vals = pd.Series(index=df.index, dtype=float)
    reversal_strength_vals = pd.Series(index=df.index, dtype=float)
    
    for i in range(10, len(df)):
        vol_ent = volume_entropy(df['volume'].iloc[:i+1])
        volume_entropy_vals.iloc[i] = vol_ent
        reversal_strength_vals.iloc[i] = price_reversal_strength(
            df['close'].iloc[:i+1], vol_ent
        )
    
    # 3. Combine factors with dynamic weighting
    alpha_factor = pd.Series(index=df.index, dtype=float)
    
    for i in range(20, len(df)):
        if (pd.notna(fractal_efficiency.iloc[i]) and 
            pd.notna(momentum_scores.iloc[i]) and 
            pd.notna(reversal_strength_vals.iloc[i])):
            
            # Dynamic weighting based on market conditions
            recent_volatility = df['close'].iloc[i-10:i+1].pct_change().std()
            vol_weight = min(recent_volatility * 100, 1.0) if not np.isnan(recent_volatility) else 0.5
            
            # Combine factors
            fractal_momentum = fractal_efficiency.iloc[i] * momentum_scores.iloc[i]
            volume_reversal = reversal_strength_vals.iloc[i] * (1 - volume_entropy_vals.iloc[i])
            
            # Final alpha factor
            alpha_factor.iloc[i] = (fractal_momentum * (1 - vol_weight) + 
                                  volume_reversal * vol_weight)
    
    # Fill NaN values with forward fill, then backward fill
    alpha_factor = alpha_factor.ffill().bfill().fillna(0)
    
    # Standardize the factor
    if alpha_factor.std() > 0:
        alpha_factor = (alpha_factor - alpha_factor.mean()) / alpha_factor.std()
    
    return alpha_factor
