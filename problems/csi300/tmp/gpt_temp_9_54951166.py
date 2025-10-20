import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    # Calculate Momentum with Decay
    def calculate_decayed_momentum(close, window=20, decay_factor=0.9):
        returns = close.pct_change()
        weights = np.array([decay_factor ** i for i in range(window)])[::-1]
        weights = weights / weights.sum()
        
        decayed_momentum = returns.rolling(window=window).apply(
            lambda x: np.dot(x, weights), raw=True
        )
        return decayed_momentum
    
    # Assess Momentum Quality
    def calculate_momentum_quality(close, volume, window=10):
        # Momentum smoothness (lower volatility of momentum changes)
        momentum = close.pct_change()
        momentum_smoothness = 1 / (momentum.rolling(window=window).std() + 1e-8)
        
        # Momentum consistency across timeframes
        short_momentum = close.pct_change(5)
        medium_momentum = close.pct_change(10)
        long_momentum = close.pct_change(20)
        momentum_consistency = (short_momentum * medium_momentum * long_momentum).abs()
        
        # Volume trend alignment
        volume_trend = volume.rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
        )
        price_trend = close.rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0], raw=False
        )
        volume_alignment = np.sign(volume_trend * price_trend)
        
        # Volume stability (inverse of volume volatility)
        volume_stability = 1 / (volume.pct_change().rolling(window=window).std() + 1e-8)
        
        # Volume rate of change
        volume_roc = volume.pct_change(window)
        
        # Composite quality score
        quality_score = (
            momentum_smoothness.rank(pct=True) * 0.3 +
            momentum_consistency.rank(pct=True) * 0.3 +
            volume_alignment.rank(pct=True) * 0.2 +
            volume_stability.rank(pct=True) * 0.1 +
            volume_roc.rank(pct=True) * 0.1
        )
        
        return quality_score
    
    # Measure Volatility Regime
    def calculate_volatility_regime(df, window=20):
        # High-Low range volatility
        hl_range = (df['high'] - df['low']) / df['close']
        hl_volatility = hl_range.rolling(window=window).std()
        
        # Returns volatility
        returns_volatility = df['close'].pct_change().rolling(window=window).std()
        
        # Combined volatility measure
        combined_volatility = (hl_volatility.rank(pct=True) * 0.5 + 
                              returns_volatility.rank(pct=True) * 0.5)
        
        # Identify regimes (1 for high volatility, 0.5 for normal, 0 for low)
        volatility_regime = pd.cut(combined_volatility, 
                                  bins=[0, 0.3, 0.7, 1], 
                                  labels=[0, 0.5, 1]).astype(float)
        
        return volatility_regime
    
    # Volume acceleration
    def calculate_volume_acceleration(volume, window=5):
        volume_ma_short = volume.rolling(window=window).mean()
        volume_ma_long = volume.rolling(window=window*2).mean()
        volume_acceleration = (volume_ma_short / volume_ma_long - 1)
        return volume_acceleration
    
    # Main factor calculation
    close = df['close']
    volume = df['volume']
    
    # Calculate components
    decayed_momentum = calculate_decayed_momentum(close)
    momentum_quality = calculate_momentum_quality(close, volume)
    volatility_regime = calculate_volatility_regime(df)
    volume_acceleration = calculate_volume_acceleration(volume)
    
    # Combine components with regime adjustment
    base_factor = decayed_momentum * momentum_quality
    
    # Regime adjustment: amplify in high volatility, dampen in low volatility
    regime_adjustment = 1 + volatility_regime * 0.5  # 1.0 to 1.5 multiplier
    
    # Volume acceleration filtering
    volume_filter = np.where(volume_acceleration > 0, 1.2, 0.8)
    
    # Final factor
    factor = base_factor * regime_adjustment * volume_filter
    
    return pd.Series(factor, index=df.index)
