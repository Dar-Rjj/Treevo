import pandas as pd
import numpy as np
import pandas as pd
import numpy as np

def heuristics_v2(df):
    """
    Volatility Regime Adjusted Price Momentum factor
    Combines price momentum with volatility regime analysis and volume confirmation
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # 1. Calculate Rolling Price Momentum
    # Short-term momentum (5-day return)
    short_momentum = data['close'].pct_change(periods=5)
    
    # Medium-term momentum (20-day return)
    medium_momentum = data['close'].pct_change(periods=20)
    
    # Combined momentum score (weighted average)
    momentum_score = 0.6 * short_momentum + 0.4 * medium_momentum
    
    # 2. Assess Volatility Regime
    # Calculate daily range (High - Low)
    daily_range = data['high'] - data['low']
    
    # Calculate rolling volatility (20-day standard deviation of daily range)
    rolling_volatility = daily_range.rolling(window=20, min_periods=10).std()
    
    # Calculate volatility percentile (50-day lookback)
    vol_percentile = rolling_volatility.rolling(window=50, min_periods=25).apply(
        lambda x: (x.iloc[-1] > np.percentile(x.dropna(), 70)) if len(x.dropna()) > 0 else 0, 
        raw=False
    )
    
    # Classify volatility regime (1 for high volatility, 0 for low volatility)
    volatility_regime = (vol_percentile > 0.5).astype(int)
    
    # 3. Adjust Momentum Based on Regime
    # Volatility adjustment factors
    high_vol_adjustment = 0.7  # Dampen in high volatility
    low_vol_adjustment = 1.3   # Amplify in low volatility
    
    # Apply regime-based adjustment
    regime_adjusted_momentum = momentum_score.copy()
    regime_adjusted_momentum[volatility_regime == 1] *= high_vol_adjustment
    regime_adjusted_momentum[volatility_regime == 0] *= low_vol_adjustment
    
    # 4. Combine with Volume Confirmation
    # Calculate volume z-score (20-day rolling)
    volume_zscore = (data['volume'] - data['volume'].rolling(window=20, min_periods=10).mean()) / \
                   data['volume'].rolling(window=20, min_periods=10).std()
    
    # Volume confirmation factor (boost momentum when volume supports the move)
    volume_confirmation = 1 + 0.2 * np.tanh(volume_zscore * np.sign(regime_adjusted_momentum))
    
    # Final factor: Volume-weighted regime adjusted momentum
    final_factor = regime_adjusted_momentum * volume_confirmation
    
    # Handle NaN values
    final_factor = final_factor.fillna(0)
    
    return final_factor
