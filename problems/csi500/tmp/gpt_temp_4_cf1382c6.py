import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility-Normalized Momentum with Dynamic Volume Confirmation
    Combines price momentum normalized by adaptive volatility with volume-based confirmation
    """
    # Extract price data
    high = df['high']
    low = df['low']
    close = df['close']
    volume = df['volume']
    
    # Core Momentum Component
    def calculate_momentum(close, periods=[5, 10, 20]):
        """Calculate multi-period momentum"""
        momentum_signals = []
        for period in periods:
            momentum = (close - close.shift(period)) / close.shift(period)
            momentum_signals.append(momentum)
        # Combine using equal weights
        combined_momentum = sum(momentum_signals) / len(momentum_signals)
        return combined_momentum
    
    raw_momentum = calculate_momentum(close)
    
    # Dynamic Volatility Estimation
    def calculate_adaptive_volatility(high, low, close, span=20):
        """Calculate exponentially weighted true range volatility"""
        # True Range calculation
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Apply exponential weighting for adaptive volatility
        volatility = true_range.ewm(span=span, adjust=False).mean()
        return volatility
    
    adaptive_vol = calculate_adaptive_volatility(high, low, close)
    
    # Volatility-normalized momentum
    volatility_normalized_momentum = raw_momentum / (adaptive_vol + 1e-8)
    
    # Volume Confirmation Module
    def calculate_volume_confidence(volume, momentum, window=20):
        """Calculate volume-based confidence score"""
        # Volume surge indicator
        volume_quantile = volume.rolling(window=window, min_periods=10).quantile(0.7)
        volume_surge = volume / (volume_quantile + 1e-8)
        
        # Nonlinear scaling using sigmoid
        volume_strength = 1 / (1 + np.exp(-2 * (volume_surge - 1)))
        
        # Directional alignment adjustment
        momentum_direction = np.sign(momentum)
        alignment_score = (momentum_direction * volume_strength + 1) / 2
        
        # Final confidence score
        confidence_score = volume_strength * alignment_score
        
        return confidence_score
    
    volume_confidence = calculate_volume_confidence(volume, raw_momentum)
    
    # Signal Integration
    def integrate_signals(vol_norm_momentum, volume_conf):
        """Combine momentum and volume components multiplicatively"""
        # Multiplicative combination
        combined_signal = vol_norm_momentum * volume_conf
        
        # Dynamic thresholding using rolling percentiles
        signal_strength = combined_signal.rolling(window=50, min_periods=20).apply(
            lambda x: (x.iloc[-1] - x.quantile(0.3)) / (x.quantile(0.7) - x.quantile(0.3) + 1e-8)
        )
        
        return signal_strength
    
    final_alpha = integrate_signals(volatility_normalized_momentum, volume_confidence)
    
    return final_alpha
