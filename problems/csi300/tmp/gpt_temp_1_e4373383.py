import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Dynamic Regime-Adaptive Acceleration Divergence factor
    Combines price acceleration with volume-price regime alignment for dynamic market state detection
    """
    
    # Compute Price Acceleration Component
    def calculate_acceleration_component(df):
        # Short-term momentum (5-day)
        short_momentum = df['close'] - df['close'].shift(5)
        
        # Medium-term momentum (20-day)
        medium_momentum = df['close'] - df['close'].shift(20)
        
        # Acceleration signal
        acceleration = short_momentum - medium_momentum
        
        # Daily price range for scaling
        daily_range = df['high'] - df['low']
        daily_range = daily_range.replace(0, np.nan)  # Avoid division by zero
        
        # Scaled acceleration
        scaled_acceleration = acceleration / daily_range
        
        return scaled_acceleration
    
    # Assess Volume-Price Regime Alignment
    def calculate_regime_alignment(df):
        # Price deviation component
        price_median = df['close'].rolling(window=20, min_periods=10).median()
        price_deviation = (df['close'] - price_median) / price_median
        
        # Volume liquidity proxy
        liquidity_ratio = df['volume'] / (df['high'] - df['low']).replace(0, np.nan)
        liquidity_ma = liquidity_ratio.rolling(window=10, min_periods=5).mean()
        liquidity_deviation = (liquidity_ratio - liquidity_ma) / liquidity_ma
        
        # Generate regime alignment score
        alignment_raw = price_deviation * liquidity_deviation
        
        # Cumulative alignment with decay factor (0.95)
        alignment_cumulative = alignment_raw.ewm(alpha=0.05, adjust=False).mean()
        
        # Scale by recent price volatility (10-day std)
        price_volatility = df['close'].pct_change().rolling(window=10, min_periods=5).std()
        
        # Final aligned score
        regime_alignment = alignment_cumulative / price_volatility.replace(0, np.nan)
        
        return regime_alignment
    
    # Dynamic Regime Detection and Weighting
    def apply_regime_weighting(df, acceleration, alignment):
        # Volatility regime detection
        returns = df['close'].pct_change()
        vol_20d = returns.rolling(window=20, min_periods=10).std()
        vol_median = vol_20d.rolling(window=60, min_periods=30).median()
        
        # High volatility regime (above median + 0.5 std)
        high_vol_threshold = vol_median + 0.5 * vol_20d.rolling(window=60, min_periods=30).std()
        high_vol_regime = vol_20d > high_vol_threshold
        
        # Trend regime detection
        acceleration_persistence = acceleration.rolling(window=10, min_periods=5).apply(
            lambda x: np.sum(x > 0) / len(x) if len(x) > 0 else 0.5
        )
        strong_trend_regime = acceleration_persistence > 0.7
        
        # Regime-adaptive weighting
        regime_weights = pd.Series(0.5, index=df.index)  # Default balanced weight
        
        # High volatility: emphasize mean reversion (alignment component)
        regime_weights[high_vol_regime] = 0.7
        
        # Strong trend: emphasize momentum (acceleration component)
        regime_weights[strong_trend_regime & ~high_vol_regime] = 0.3
        
        # Transition regimes (neither strong trend nor high vol): balanced
        transition_regime = ~strong_trend_regime & ~high_vol_regime
        regime_weights[transition_regime] = 0.5
        
        return regime_weights
    
    # Main factor calculation
    acceleration = calculate_acceleration_component(df)
    alignment = calculate_regime_alignment(df)
    regime_weights = apply_regime_weighting(df, acceleration, alignment)
    
    # Combine components with regime-adaptive weighting
    momentum_component = acceleration * (1 - regime_weights)
    mean_reversion_component = alignment * regime_weights
    
    # Final alpha signal
    alpha_signal = momentum_component + mean_reversion_component
    
    # Normalize by recent volatility for better comparability
    recent_vol = df['close'].pct_change().rolling(window=20, min_periods=10).std()
    normalized_alpha = alpha_signal / recent_vol.replace(0, np.nan)
    
    return normalized_alpha
